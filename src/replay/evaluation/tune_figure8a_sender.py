"""只在 Cityscapes train 上选择图 8(a) 的发送端背景低通强度。"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from replay.core.assets import verify_model_lock
from replay.core.config import ReplayConfig, write_json
from replay.data.cityscapes import cityscapes_pairs, load_cityscapes_pair
from replay.evaluation.figure8 import binary_counts, ground_truth_patches
from replay.transmission.header import FrameHeader, SessionHeader
from replay.transmission.jpeg import encode_under_budget
from replay.transmission.packet_loss import plan_entropy_packets
from replay.transmission.packet_wire import (CONTROL_FIXED_BYTES, INTERVAL_MANIFEST_BYTES,
                                             pack_packetized_frame)


TARGET_RAW_IDS = {"car": (13, 26), "pedestrian": (11, 24), "vegetation": (8, 21)}
PACKET_PAYLOAD_BYTES = 1200
PACKET_HEADER_BYTES = 8
RESTART_BLOCKS = 2048


def select_train_positive_pairs(root: Path, count: int, seed: int
                                ) -> dict[str, list[tuple[Path, Path]]]:
    """固定种子扫描 train 标签；只返回确有目标像素的图像。"""
    if count < 1:
        raise ValueError("每类图像数必须为正")
    pairs = list(cityscapes_pairs(root, "train"))
    if not pairs:
        raise ValueError("Cityscapes train 为空")
    random.Random(seed).shuffle(pairs)
    selected: dict[str, list[tuple[Path, Path]]] = {name: [] for name in TARGET_RAW_IDS}
    for pair in pairs:
        with Image.open(pair[1]) as source:
            raw = np.asarray(source, dtype=np.uint8)
        for name, (_, raw_id) in TARGET_RAW_IDS.items():
            if len(selected[name]) < count and np.any(raw == raw_id):
                selected[name].append(pair)
        if all(len(value) == count for value in selected.values()):
            break
    short = {name: len(value) for name, value in selected.items() if len(value) < count}
    if short:
        raise ValueError(f"Cityscapes train 的目标类别样本不足：{short}")
    # 与正式验证入口一样，B 只在各任务的第一帧计费。
    return {name: sorted(value) for name, value in selected.items()}


def encode_wire_under_budget(image: Image.Image, focus: np.ndarray,
                             config: ReplayConfig, *, task: str, sigma: float,
                             budget_bytes: int, first_frame: bool
                             ) -> tuple[bytes, int, dict]:
    """完整编码并实际封包；以最终 wire 字节而非 JPEG 文件大小验预算。"""
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
        raise ValueError("控制头已耗尽 30 kB 预算")
    payload, info = encode_under_budget(
        image, focus, patch, remaining, 0,
        int(importance["dct_high_frequency_cutoff"]),
        luma_strength=float(importance["prequant_luma_strength"]),
        chroma_strength=float(importance["prequant_chroma_strength"]),
        gamma=float(importance["prequant_gamma"]),
        importance_ratio_low=float(importance["importance_ratio_low"]),
        importance_ratio_high=float(importance["importance_ratio_high"]),
        restart_marker_blocks=RESTART_BLOCKS,
        packet_header_bytes=PACKET_HEADER_BYTES,
        packet_manifest_bytes_per_interval=INTERVAL_MANIFEST_BYTES,
        optimize_huffman=True, background_lowpass_sigma=sigma,
        fast_prequant=True)
    width, height, intervals, packets = plan_entropy_packets(payload, PACKET_PAYLOAD_BYTES)
    if (width, height) != image.size:
        raise AssertionError("发送端改变了原始图像尺寸")
    wire = pack_packetized_frame(
        session, frame, payload, intervals, packets,
        packet_payload_bytes=PACKET_PAYLOAD_BYTES,
        packet_header_bytes=PACKET_HEADER_BYTES)
    expected = (CONTROL_FIXED_BYTES + len(session or b"") + len(frame)
                + info["codec_wire_bytes"])
    if wire.sent_bytes != expected or wire.sent_bytes > budget_bytes:
        raise AssertionError("发送端实际 wire 字节与预算搜索不一致")
    return payload, wire.sent_bytes, info


def summarize_sender_rows(rows: list[dict], tasks: tuple[str, ...],
                          sigmas: tuple[float, ...], expected_per_task: int) -> dict:
    """不可行图像记 TP=FP=0、FN=目标像素数，保证候选用同一完整样本池。"""
    summary: dict[str, dict] = {}
    for task in tasks:
        candidates = {}
        for sigma in sigmas:
            selected = [row for row in rows if row["task"] == task and row["sigma"] == sigma]
            if len(selected) != expected_per_task:
                raise ValueError(f"{task}/sigma={sigma} 仅 {len(selected)}/{expected_per_task} 张")
            if len({row["image_id"] for row in selected}) != expected_per_task:
                raise ValueError("参数选择中出现重复图像")
            total = np.asarray([row["counts"] for row in selected], dtype=np.int64).sum(axis=0)
            denominator = int(total.sum())
            score = float(total[0] / denominator) if denominator else None
            successes = [row for row in selected if row["status"] == "ok"]
            candidates[str(sigma)] = {
                "iou_with_failures_zero": score,
                "counts_tp_fp_fn": total.tolist(),
                "feasible_images": len(successes),
                "feasible_rate": len(successes) / expected_per_task,
                "mean_wire_bytes_success": (float(np.mean([row["sent_bytes"] for row in successes]))
                                             if successes else None),
                "median_quality_success": (float(np.median([row["jpeg_quality"] for row in successes]))
                                           if successes else None),
            }
        # 相同 IoU 时取更弱的低通，以免为了码率过度抹平背景。
        ranked = sorted(sigmas, key=lambda value: (
            -(candidates[str(value)]["iou_with_failures_zero"] or 0.0), value))
        summary[task] = {"selected_sigma": ranked[0], "candidates": candidates}
    return summary


def tune_figure8a_sender(config: ReplayConfig, device: torch.device, *,
                         output: str | Path, sample_images_per_task: int = 50,
                         sigmas: tuple[float, ...] = (16.0, 32.0, 64.0),
                         budget_bytes: int = 30_000, seed: int = 2026) -> dict:
    """训练集选择；正式图8(a)入口只能读取确定的参数，不从 val 回调调参。"""
    if not sigmas or len(set(sigmas)) != len(sigmas) or any(value <= 0 for value in sigmas):
        raise ValueError("sigma 候选必须是互不相同的正数")
    if budget_bytes <= 0:
        raise ValueError("预算必须为正")
    pairs = select_train_positive_pairs(config.data_path("cityscapes_root"),
                                        sample_images_per_task, seed)
    verify_model_lock(config, "segmenter")
    output = config.path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_cfg = config.section("image")
    settings = {
        "protocol": "figure8a_sender_train_tuning_v1",
        "split": "train", "pilot": sample_images_per_task < 50,
        "sample_images_per_task": sample_images_per_task,
        "tasks": list(TARGET_RAW_IDS), "sigmas": list(sigmas),
        "budget_wire_bytes": budget_bytes, "seed": seed,
        "sampled_image_ids": {name: [pair[0].stem for pair in values]
                              for name, values in pairs.items()},
        "native_resolution": [2048, 1024], "jpeg_subsampling": "4:4:4",
        "optimize_huffman": True, "restart_blocks": RESTART_BLOCKS,
        "fast_prequant": True, "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
        "packet_header_bytes": PACKET_HEADER_BYTES,
        "segmenter_revision": config.section("models")["segmenter"]["revision"],
        "config_sha256": sha256(json.dumps(config.values, sort_keys=True,
                                            default=str).encode()).hexdigest(),
    }
    settings_path = output / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("已有参数选择目录使用了不同设置")
    else:
        write_json(settings_path, settings)
    runs_path = output / "runs.jsonl"
    completed: dict[tuple[str, float, str], dict] = {}
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = row["task"], float(row["sigma"]), row["image_id"]
                if key in completed:
                    raise ValueError(f"重复的训练集调参记录：{key}")
                completed[key] = row

    from replay.evaluation.segmentation import CityscapesSegmenter
    segmenter = CityscapesSegmenter(config.model_path("segmenter"), device,
                                    int(image_cfg["segformer_tile"]),
                                    int(image_cfg["segformer_stride"]))
    for task, task_pairs in pairs.items():
        class_id = TARGET_RAW_IDS[task][0]
        for index, pair in enumerate(task_pairs):
            image_id = pair[0].stem
            if all((task, sigma, image_id) in completed for sigma in sigmas):
                continue
            image, truth = load_cityscapes_pair(*pair)
            if image.size != (2048, 1024):
                raise ValueError(f"非原生 Cityscapes 分辨率：{pair[0]}")
            focus = ground_truth_patches(truth, class_id, int(image_cfg["patch_size"]))
            gt_pixels = int((truth == class_id).sum())
            if gt_pixels == 0:
                raise AssertionError("训练集采样选到了不含目标类别的图像")
            for sigma in sigmas:
                key = task, sigma, image_id
                if key in completed:
                    continue
                try:
                    payload, sent_bytes, info = encode_wire_under_budget(
                        image, focus, config, task=task, sigma=sigma,
                        budget_bytes=budget_bytes, first_frame=index == 0)
                except ValueError as error:
                    if "预算" not in str(error) and "无法满足" not in str(error):
                        raise
                    row = {"task": task, "class_id": class_id, "image_id": image_id,
                           "sigma": sigma, "status": "infeasible",
                           "counts": [0, 0, gt_pixels], "sent_bytes": None,
                           "jpeg_quality": None, "reason": str(error)}
                else:
                    with Image.open(BytesIO(payload)) as decoded:
                        decoded.load()
                        candidate = decoded.convert("RGB")
                    counts = binary_counts(segmenter.predict(candidate), truth, class_id)
                    row = {"task": task, "class_id": class_id, "image_id": image_id,
                           "sigma": sigma, "status": "ok", "counts": list(counts),
                           "sent_bytes": sent_bytes,
                           "jpeg_quality": int(info["jpeg_quality"]), "reason": None}
                with runs_path.open("a", encoding="utf-8") as destination:
                    destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                completed[key] = row
            image.close()
            if (index + 1) % 10 == 0:
                print(f"[sender tuning {task} {index + 1}/{len(task_pairs)}]",
                      flush=True)

    rows = list(completed.values())
    summary = summarize_sender_rows(rows, tuple(TARGET_RAW_IDS), sigmas,
                                    sample_images_per_task)
    report = {"settings": settings, "summary": summary, "runs": rows}
    write_json(output / "report.json", report)
    return report
