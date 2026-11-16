"""Cityscapes train 上的 30 kB 原尺寸 DCT 预量化消融；不读取 val。"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from replay.core.assets import verify_model_lock
from replay.core.config import ReplayConfig, write_json
from replay.data.cityscapes import load_cityscapes_pair
from replay.evaluation.figure8 import binary_counts, ground_truth_patches
from replay.evaluation.tune_figure8a_sender import (
    PACKET_HEADER_BYTES, PACKET_PAYLOAD_BYTES, RESTART_BLOCKS,
    TARGET_RAW_IDS, select_train_positive_pairs,
)
from replay.transmission.header import FrameHeader, SessionHeader
from replay.transmission.jpeg import encode_under_budget
from replay.transmission.packet_loss import plan_entropy_packets
from replay.transmission.packet_wire import (
    CONTROL_FIXED_BYTES, INTERVAL_MANIFEST_BYTES, pack_packetized_frame,
)


# 只改变发送端未公开的 DCT 抑制强度；blur16 是现行方案的训练集对照。
DEFAULT_CANDIDATES = (
    {"id": "no_blur_420_halo8", "sigma": 0.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 2,
     "restart_blocks": 512, "roi_halo_pixels": 8.0},
    {"id": "plain2", "sigma": 0.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 0,
     "restart_blocks": 2048, "roi_halo_pixels": 0.0},
    {"id": "dct8", "sigma": 0.0, "strength": 8.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 0,
     "restart_blocks": 2048, "roi_halo_pixels": 0.0},
    {"id": "dct16", "sigma": 0.0, "strength": 16.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 0,
     "restart_blocks": 2048, "roi_halo_pixels": 0.0},
    {"id": "dct16_zero1", "sigma": 0.0, "strength": 16.0, "cutoff": 2,
     "zero_threshold_multiplier": 1.0, "subsampling": 0,
     "restart_blocks": 2048, "roi_halo_pixels": 0.0},
    {"id": "dct32_zero1", "sigma": 0.0, "strength": 32.0, "cutoff": 2,
     "zero_threshold_multiplier": 1.0, "subsampling": 0,
     "restart_blocks": 2048, "roi_halo_pixels": 0.0},
    {"id": "blur16_control", "sigma": 16.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 0,
     "restart_blocks": 2048, "roi_halo_pixels": 0.0},
    {"id": "blur16_420", "sigma": 16.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 2,
     "restart_blocks": 512, "roi_halo_pixels": 0.0},
    {"id": "blur32_420", "sigma": 32.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 2,
     "restart_blocks": 512, "roi_halo_pixels": 0.0},
    {"id": "blur64_420", "sigma": 64.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 2,
     "restart_blocks": 512, "roi_halo_pixels": 0.0},
    {"id": "blur16_420_halo8", "sigma": 16.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 2,
     "restart_blocks": 512, "roi_halo_pixels": 8.0},
    {"id": "blur32_420_halo8", "sigma": 32.0, "strength": 2.0, "cutoff": 2,
     "zero_threshold_multiplier": 0.0, "subsampling": 2,
     "restart_blocks": 512, "roi_halo_pixels": 8.0},
)


def _validate_candidates(candidates: tuple[dict, ...]) -> None:
    if not candidates or len({item["id"] for item in candidates}) != len(candidates):
        raise ValueError("候选参数不能为空且 id 不得重复")
    for item in candidates:
        if (not item["id"] or item["sigma"] < 0 or item["strength"] < 0 or
                not 0 <= item["cutoff"] <= 14 or
                item["zero_threshold_multiplier"] < 0 or
                item["subsampling"] not in (0, 2) or
                item["restart_blocks"] < 1 or
                item["roi_halo_pixels"] < 0 or
                not all(np.isfinite(item[key]) for key in
                        ("sigma", "strength", "zero_threshold_multiplier",
                         "roi_halo_pixels"))):
            raise ValueError(f"DCT 候选参数无效：{item}")


def encode_dct_candidate(image: Image.Image, focus: np.ndarray,
                         config: ReplayConfig, *, task: str, candidate: dict,
                         budget_bytes: int, first_frame: bool
                         ) -> tuple[bytes, int, dict]:
    """JPEG 像素尺寸不变，按实际控制清单/分片头/JPEG 总字节验 30 kB。"""
    _validate_candidates((candidate,))
    if task not in TARGET_RAW_IDS or budget_bytes <= 0:
        raise ValueError("任务或预算无效")
    image_cfg = config.section("image")
    importance = config.section("importance")
    patch = int(image_cfg["patch_size"])
    clip = config.section("models")["clip"]
    session = (SessionHeader(task, (task,), clip["repo"], clip["revision"]).to_bytes()
               if first_frame else None)
    frame = FrameHeader(image.width, image.height, patch, focus).to_bytes()
    remaining = budget_bytes - CONTROL_FIXED_BYTES - len(session or b"") - len(frame)
    if remaining <= 0:
        raise ValueError("控制头已耗尽传输预算")
    payload, info = encode_under_budget(
        image, focus, patch, remaining, candidate["subsampling"],
        candidate["cutoff"],
        luma_strength=candidate["strength"],
        chroma_strength=candidate["strength"],
        gamma=float(importance["prequant_gamma"]),
        importance_ratio_low=float(importance["importance_ratio_low"]),
        importance_ratio_high=float(importance["importance_ratio_high"]),
        background_lowpass_sigma=candidate["sigma"], fast_prequant=True,
        zero_threshold_multiplier=candidate["zero_threshold_multiplier"],
        roi_halo_pixels=candidate["roi_halo_pixels"],
        allow_420=candidate["subsampling"] == 2,
        restart_marker_blocks=candidate["restart_blocks"],
        packet_header_bytes=PACKET_HEADER_BYTES,
        packet_manifest_bytes_per_interval=INTERVAL_MANIFEST_BYTES,
        optimize_huffman=True)
    width, height, intervals, packets = plan_entropy_packets(
        payload, PACKET_PAYLOAD_BYTES,
        allow_420=candidate["subsampling"] == 2)
    if (width, height) != image.size:
        raise AssertionError("DCT 消融改变了原始图像尺寸")
    wire = pack_packetized_frame(
        session, frame, payload, intervals, packets,
        packet_payload_bytes=PACKET_PAYLOAD_BYTES,
        packet_header_bytes=PACKET_HEADER_BYTES)
    if (wire.sent_bytes != CONTROL_FIXED_BYTES + len(session or b"") +
            len(frame) + info["codec_wire_bytes"] or
            wire.sent_bytes > budget_bytes):
        raise AssertionError("DCT 消融的实际 wire 字节与预算不一致")
    return payload, wire.sent_bytes, info


def summarize_dct_rows(rows: list[dict], candidates: tuple[dict, ...],
                       sample_count: int) -> dict:
    """同一完整样本池计分；无法编码的图像作为目标类漏检。"""
    result = {}
    for task in TARGET_RAW_IDS:
        options = {}
        for candidate in candidates:
            selected = [row for row in rows if row["task"] == task and
                        row["candidate_id"] == candidate["id"]]
            if len(selected) != sample_count or len({row["image_id"] for row in selected}) != sample_count:
                raise ValueError(f"{task}/{candidate['id']} 的训练集记录不完整")
            counts = np.asarray([row["counts"] for row in selected], dtype=np.int64).sum(axis=0)
            feasible = [row for row in selected if row["status"] == "ok"]
            options[candidate["id"]] = {
                "iou_with_failures_zero": (float(counts[0] / counts.sum())
                                           if counts.sum() else None),
                "counts_tp_fp_fn": counts.tolist(),
                "feasible_images": len(feasible),
                "feasible_rate": len(feasible) / sample_count,
                "median_quality_success": (float(np.median([r["jpeg_quality"] for r in feasible]))
                                           if feasible else None),
                "candidate": candidate,
            }
        ranked = sorted(candidates, key=lambda item: (
            -(options[item["id"]]["iou_with_failures_zero"] or 0.0),
            item["sigma"], item["strength"], item["zero_threshold_multiplier"]))
        result[task] = {"selected_candidate_id": ranked[0]["id"],
                        "candidates": options}
    return result


def run_dct_ablation(config: ReplayConfig, device: torch.device, *,
                     output: str | Path, sample_images_per_task: int = 20,
                     candidate_ids: tuple[str, ...] | None = None,
                     budget_bytes: int = 30_000, seed: int = 2027) -> dict:
    """仅使用 Cityscapes train 的 GT 目标区域；无线丢失另由固定评测验证。"""
    candidates = tuple(item for item in DEFAULT_CANDIDATES
                       if candidate_ids is None or item["id"] in candidate_ids)
    _validate_candidates(candidates)
    if candidate_ids is not None and len(candidates) != len(set(candidate_ids)):
        raise ValueError("未知或重复的候选 id")
    pairs = select_train_positive_pairs(config.data_path("cityscapes_root"),
                                        sample_images_per_task, seed)
    verify_model_lock(config, "segmenter")
    output = config.path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_cfg = config.section("image")
    settings = {
        "protocol": "figure8a_train_dct_ablation_v1",
        "split": "train", "pilot": sample_images_per_task < 50,
        "sample_images_per_task": sample_images_per_task,
        "sampled_image_ids": {name: [pair[0].stem for pair in selected]
                              for name, selected in pairs.items()},
        "candidates": list(candidates), "budget_wire_bytes": budget_bytes,
        "seed": seed, "native_resolution": [2048, 1024],
        "jpeg_subsampling": "candidate-dependent 4:4:4 or 4:2:0",
        "optimize_huffman": True,
        "restart_blocks": "candidate-dependent 2048 or 512 MCU",
        "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
        "packet_header_bytes": PACKET_HEADER_BYTES,
        "no_loss_codec_only": True,
        "failure_policy": "infeasible counts as TP=FP=0,FN=GT target pixels",
        "segmenter_revision": config.section("models")["segmenter"]["revision"],
        "config_sha256": sha256(json.dumps(config.values, sort_keys=True,
                                            default=str).encode()).hexdigest(),
    }
    settings_path = output / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("输出目录已有不同的 DCT 消融配置")
    else:
        write_json(settings_path, settings)
    runs_path = output / "runs.jsonl"
    completed: dict[tuple[str, str, str], dict] = {}
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = row["task"], row["candidate_id"], row["image_id"]
                if key in completed:
                    raise ValueError(f"重复的 DCT 消融记录：{key}")
                completed[key] = row
    from replay.evaluation.segmentation import CityscapesSegmenter
    segmenter = CityscapesSegmenter(
        config.model_path("segmenter"), device,
        int(image_cfg["segformer_tile"]), int(image_cfg["segformer_stride"]))
    for task, selected_pairs in pairs.items():
        class_id = TARGET_RAW_IDS[task][0]
        for index, pair in enumerate(selected_pairs):
            image_id = pair[0].stem
            if all((task, item["id"], image_id) in completed for item in candidates):
                continue
            image, truth = load_cityscapes_pair(*pair)
            if image.size != (2048, 1024):
                raise ValueError(f"非原生 Cityscapes 尺寸：{image.size}")
            focus = ground_truth_patches(truth, class_id, int(image_cfg["patch_size"]))
            gt_pixels = int((truth == class_id).sum())
            for candidate in candidates:
                key = task, candidate["id"], image_id
                if key in completed:
                    continue
                try:
                    payload, sent_bytes, info = encode_dct_candidate(
                        image, focus, config, task=task, candidate=candidate,
                        budget_bytes=budget_bytes, first_frame=index == 0)
                except ValueError as error:
                    if "预算" not in str(error) and "无法满足" not in str(error):
                        raise
                    row = {"task": task, "class_id": class_id, "image_id": image_id,
                           "candidate_id": candidate["id"], "status": "infeasible",
                           "counts": [0, 0, gt_pixels], "sent_bytes": None,
                           "jpeg_quality": None, "reason": str(error)}
                else:
                    with Image.open(BytesIO(payload)) as decoded:
                        candidate_image = decoded.convert("RGB")
                    counts = binary_counts(segmenter.predict(candidate_image), truth, class_id)
                    row = {"task": task, "class_id": class_id, "image_id": image_id,
                           "candidate_id": candidate["id"], "status": "ok",
                           "counts": list(counts), "sent_bytes": sent_bytes,
                           "jpeg_quality": int(info["jpeg_quality"]), "reason": None}
                with runs_path.open("a", encoding="utf-8") as destination:
                    destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                completed[key] = row
            image.close()
            if (index + 1) % 5 == 0:
                print(f"[DCT ablation {task} {index + 1}/{len(selected_pairs)}]",
                      flush=True)
    rows = list(completed.values())
    report = {"settings": settings,
              "summary": summarize_dct_rows(rows, candidates, sample_images_per_task),
              "runs": rows}
    write_json(output / "report.json", report)
    return report
