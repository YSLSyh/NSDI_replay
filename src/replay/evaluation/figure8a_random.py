"""Cityscapes val 上的单 JPEG KDC 与 LimitNet 按类别 IoU 对照。

两条码流使用相同的字节预算、1200 B 数据分片和按图像 ID 固定的
Gilbert–Elliott 丢包事件前缀。JPEG 无 RST；这不是实测 5G 轨迹。
"""

from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from replay.core.assets import verify_model_lock
from replay.core.config import ReplayConfig
from replay.data.cityscapes import cityscapes_pairs, load_cityscapes_pair
from replay.evaluation.figure8 import binary_counts
from replay.evaluation.figure8a_packet import (
    LIMITNET_MODES, _limit_under_budget, _limitnet_input, _limitnet_output,
    _receive_limit, _validate_limitnet_checkpoint_metadata,
)
from replay.knowledge.bins import read_bins
from replay.transmission.channel import _PROFILES
from replay.transmission.header import FrameHeader, SessionHeader
from replay.transmission.packet_loss import _packet_loss_events
from replay.transmission.plain_jpeg_channel import simulate_plain_jpeg_entropy_loss
from replay.transmission.task_jpeg_search import search_task_jpeg


CLASS_IDS = {
    "car": 13, "pedestrian": 11, "traffic light": 6, "traffic sign": 7,
    "rider": 12, "train": 16, "motorcycle": 17, "bicycle": 18,
}
METHODS = ("kdc_restored", "limitnet")
PACKET_PAYLOAD_BYTES = 1200
PACKET_HEADER_BYTES = 8
SCENARIO = "scooter"
LEVEL = "medium"
MEAN_BURST_PACKETS = 3.0


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _settings_file(path: Path, settings: dict) -> str:
    """同目录可由两个 GPU 分片并行写不同图像，设置只允许初始化一次。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _json_bytes(settings)
    try:
        with path.open("xb") as handle:
            handle.write(expected)
    except FileExistsError:
        for _ in range(20):
            try:
                actual = path.read_bytes()
                if actual == expected:
                    break
                if len(actual) >= len(expected):
                    raise ValueError(f"输出目录已有不同实验设置：{path}")
            except PermissionError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(f"无法读取已创建的实验设置：{path}")
    return sha256(expected).hexdigest()


def _stage_index(config: ReplayConfig, stage12: Path) -> tuple[dict, list[str]]:
    index = json.loads((stage12 / "index.json").read_text(encoding="utf-8"))
    if index.get("method") != "region_importance_components_v6":
        raise ValueError("阶段二缓存不是 CLIPSeg 区域重要性版本")
    bins_path = stage12 / "semantic_bins.json"
    bins = read_bins(bins_path, config.section("task")["description"])
    task = config.section("task")
    cache = json.loads(bins_path.read_text(encoding="utf-8"))
    audit = cache.get("selection_audit")
    if (not 4 <= len(bins) <= 8 or len(set(bins)) != len(bins) or
            set(task.get("required_bins", [])).difference(bins) or
            set(bins).difference(task.get("allowed_bins", bins)) or
            not isinstance(audit, dict) or audit.get("selected_bins") != bins or
            (task.get("required_bins") is not None and
             audit.get("required_bins") != task["required_bins"]) or
            (task.get("allowed_bins") is not None and
             audit.get("allowed_bins") != task["allowed_bins"])):
        raise ValueError("阶段二语义桶未通过当前 4–8 类配置校验")
    if (index.get("task") != task["description"] or index.get("bins") != bins or
            index.get("completed_images") != len(index.get("samples", []))):
        raise ValueError("阶段二索引未完成或与当前任务不符")
    unsupported = set(bins).difference(CLASS_IDS)
    if unsupported:
        raise ValueError(f"语义桶没有 Cityscapes trainId 映射：{sorted(unsupported)}")
    return index, bins


def _wait_for_sample(config: ReplayConfig, stage12: Path, position: int,
                     expected_image: Path, deadline: float) -> tuple[dict, list[str], Path]:
    """允许 GPU 评测与逐图生成的 val 重要性缓存流水运行。"""
    while True:
        try:
            index, bins = _stage_index(config, stage12)
            samples = index["samples"]
            if len(samples) > position:
                sample = samples[position]
                source = Path(sample["image"])
                if source.resolve() != expected_image.resolve() or sample.get("bins") != bins:
                    raise ValueError(f"阶段二第 {position + 1} 张不是对应的 Cityscapes val 图像")
                folder = stage12 / f"{position + 1:02d}_{source.stem}"
                if ((folder / "patch_alpha_fp32.npy").is_file() and
                        (folder / "selected_patch_mask.npy").is_file()):
                    return sample, bins, folder
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                f"等待阶段二第 {position + 1} 张 val 重要性缓存超时：{stage12}")
        time.sleep(5)


def _decode_intact(payload: bytes, size: tuple[int, int]) -> Image.Image:
    with Image.open(BytesIO(payload)) as source:
        source.load()
        if source.size != size:
            raise ValueError("KDC JPEG 解码尺寸与 Cityscapes 原图不符")
        return source.convert("RGB")


def _method_path(case_dir: Path, method: str) -> Path:
    return case_dir / f"{method}.json"


def _stored(path: Path, *, settings_sha256: str, image_id: str,
            method: str) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("settings_sha256") != settings_sha256 or
            value.get("image_id") != image_id or value.get("method") != method):
        raise ValueError(f"既有逐图结果与当前实验不一致：{path}")
    return value


def _failure_counts(truth: np.ndarray, class_ids: dict[str, int]) -> dict[str, list[int]]:
    return {name: [0, 0, int(np.count_nonzero(truth == class_id))]
            for name, class_id in class_ids.items()}


def _counts(prediction: np.ndarray, truth: np.ndarray,
            class_ids: dict[str, int]) -> dict[str, list[int]]:
    return {name: list(binary_counts(prediction, truth, class_id))
            for name, class_id in class_ids.items()}


def _psnr(original: Image.Image, candidate: Image.Image) -> float | None:
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(candidate, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else None


def _is_budget_error(error: ValueError) -> bool:
    return "预算" in str(error) or "字节" in str(error)


def _kdc_case(config: ReplayConfig, image: Image.Image, truth: np.ndarray,
              image_id: str, bins: list[str], alpha: np.ndarray,
              selected: np.ndarray, *, budget_bytes: int, seed: int,
              roi_weight: float, background_weight: float,
              receiver: tuple, segmenter, class_ids: dict[str, int],
              save_image: Path | None) -> dict:
    from replay.receiver.infer import restore

    patch = int(config.section("image")["patch_size"])
    importance = config.section("importance")
    clip = config.section("models")["clip"]
    session = SessionHeader(config.section("task")["description"], tuple(bins),
                            clip["repo"], clip["revision"]).to_bytes()
    frame = FrameHeader(image.width, image.height, patch, selected).to_bytes()
    control_bytes = len(session) + len(frame)
    base = {"method": "kdc_restored", "status": "infeasible",
            "sent_bytes": None, "packet_count": 0, "lost_packets": 0,
            "lost_packet_indices": [], "receiver_applied": False,
            "semantic_control_bytes": control_bytes, "counts": _failure_counts(truth, class_ids)}
    try:
        payload, search = search_task_jpeg(
            image, alpha, budget_bytes=budget_bytes, patch_size=patch,
            selected_patch_mask=selected, roi_weight=roi_weight,
            background_weight=background_weight,
            packet_payload_bytes=PACKET_PAYLOAD_BYTES,
            packet_header_bytes=PACKET_HEADER_BYTES,
            fixed_control_bytes=control_bytes,
            max_quality=95, subsampling=int(config.section("image")["jpeg_subsampling"]),
            cutoff=int(importance["dct_high_frequency_cutoff"]),
            gamma=float(importance["prequant_gamma"]),
            importance_ratio_low=float(importance["importance_ratio_low"]),
            importance_ratio_high=float(importance["importance_ratio_high"]))
    except ValueError as error:
        if not _is_budget_error(error):
            raise
        return {**base, "reason": str(error)}
    _decode_intact(payload, image.size)
    outcome = simulate_plain_jpeg_entropy_loss(
        payload, image_id=image_id, expected_size=image.size,
        scenario=SCENARIO, level=LEVEL, seed=seed,
        packet_payload_bytes=PACKET_PAYLOAD_BYTES,
        packet_header_bytes=PACKET_HEADER_BYTES,
        mean_burst_packets=MEAN_BURST_PACKETS)
    total_sent = outcome.sent_bytes + control_bytes
    if total_sent != search["codec_wire_bytes"] or total_sent > budget_bytes:
        raise AssertionError("KDC 搜索字节账本与真实随机分包不一致")
    fields = {**base, "status": outcome.status, "sent_bytes": total_sent,
              "received_bytes": outcome.received_bytes + control_bytes,
              "jpeg_bytes": len(payload), "jpeg_quality": search["jpeg_quality"],
              "packet_count": outcome.packet_count,
              "lost_packets": outcome.lost_packets,
              "lost_packet_indices": list(outcome.lost_packet_indices),
              "actual_loss_rate": outcome.actual_loss_rate,
              "lost_entropy_bytes": outcome.lost_entropy_bytes,
              "decode_error": outcome.decode_error,
              "selected_plain_jpeg_fallback": search["plain_jpeg"],
              "sender_used_importance": not search["plain_jpeg"],
              "search_selection": search}
    if outcome.decoded is None:
        return fields
    if outcome.lost_packets:
        _, shared, kb, model, device = receiver
        restored = restore(outcome.decoded, bins, shared, kb, model,
                           int(config.section("knowledge")["rx_top_k"]), device,
                           update_session=False, missing_pixels=None)
        fields["receiver_applied"] = True
    else:
        restored = outcome.decoded
    if save_image is not None:
        restored.save(save_image)
    fields["counts"] = _counts(segmenter.predict(restored), truth, class_ids)
    fields["psnr_db"] = _psnr(image, restored)
    return fields


def _limit_case(image: Image.Image, truth: np.ndarray, image_id: str,
                *, budget_bytes: int, seed: int, codec, mode: str,
                segmenter, class_ids: dict[str, int],
                save_image: Path | None) -> dict:
    base = {"method": "limitnet", "status": "infeasible",
            "sent_bytes": None, "packet_count": 0, "lost_packets": 0,
            "lost_packet_indices": [], "layers_received": 0,
            "counts": _failure_counts(truth, class_ids)}
    limit_input = _limitnet_input(image, mode)
    try:
        wire, layers_sent = _limit_under_budget(
            codec, limit_input, None, budget_bytes, parity_packets_per_layer=0)
    except ValueError as error:
        if not _is_budget_error(error):
            raise
        return {**base, "reason": str(error)}
    rate = float(_PROFILES[SCENARIO]["loss_rates"][LEVEL])
    events, _ = _packet_loss_events(
        len(wire.packets), rate, seed, image_id, SCENARIO,
        "gilbert_elliott", None, MEAN_BURST_PACKETS)
    candidate, layers_received, received_bytes = _receive_limit(wire, events, codec)
    fields = {**base, "status": "ok" if layers_received else "undecodable",
              "sent_bytes": wire.sent_bytes, "received_bytes": received_bytes,
              "packet_count": len(events), "lost_packets": sum(events),
              "lost_packet_indices": [i for i, lost in enumerate(events) if lost],
              "actual_loss_rate": sum(events) / len(events) if events else 0.0,
              "layers_sent": layers_sent, "layers_received": layers_received,
              "codec_input_size": list(limit_input.size), "checkpoint_source": codec.source,
              "parity_packets_per_layer": 0}
    if wire.sent_bytes > budget_bytes:
        raise AssertionError("LimitNet 传输超过预算")
    if layers_received == 0:
        return fields
    candidate = _limitnet_output(candidate, image.size, mode)
    if save_image is not None:
        candidate.save(save_image)
    fields["counts"] = _counts(segmenter.predict(candidate), truth, class_ids)
    fields["psnr_db"] = _psnr(image, candidate)
    return fields


def run_random_val(config: ReplayConfig, *, stage12: str | Path,
                   output: str | Path, limitnet_checkpoint: str | Path,
                   limitnet_mode: str = "native_adapted",
                   device: torch.device, budget_bytes: int = 30_000,
                   max_images: int = 100, seed: int = 2026,
                   num_shards: int = 1, shard_index: int = 0,
                   wait_stage12_seconds: float = 0,
                   roi_weight: float = 4.0, background_weight: float = 1.0,
                   save_examples: int = 0) -> dict:
    """每图两方法独立落盘；相同目录可从 100 张续跑至 500 张。"""
    from replay.baselines.limitnet import load_limitnet
    from replay.evaluation.segmentation import CityscapesSegmenter
    from replay.knowledge.encoders import SharedKB
    from replay.knowledge.store import PosteriorKB
    from replay.receiver.infer import load_receiver

    if (limitnet_mode not in LIMITNET_MODES or budget_bytes <= 0 or
            not 1 <= max_images <= 500 or num_shards < 1 or
            not 0 <= shard_index < num_shards or wait_stage12_seconds < 0 or
            save_examples < 0 or roi_weight <= 0 or background_weight <= 0):
        raise ValueError("评测参数无效")
    pairs = list(cityscapes_pairs(config.data_path("cityscapes_root"), "val"))
    if len(pairs) != 500:
        raise ValueError(f"Cityscapes val 应有 500 张完整标注图，实际 {len(pairs)}")
    stage12, output = config.path(stage12), config.path(output)
    first, bins, _ = _wait_for_sample(
        config, stage12, 0, pairs[0][0], time.monotonic() + wait_stage12_seconds)
    del first
    class_ids = {name: CLASS_IDS[name] for name in bins}
    checkpoint = config.path(limitnet_checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    variant = _validate_limitnet_checkpoint_metadata(limitnet_mode, metadata)
    del payload
    verify_model_lock(config, "segmenter")
    receiver_path = config.receiver_checkpoint()
    kb_path = config.path(_stage_index(config, stage12)[0].get("kb_path") or
                          config.section("data")["kb_path"])
    settings = {
        "protocol": "figure8a_single_jpeg_synthetic_ge_val_v1",
        "split": "val", "n_available_val_images": 500,
        "val_image_ids_sha256": sha256("\n".join(path.stem for path, _ in pairs).encode()).hexdigest(),
        "stage12": str(stage12), "semantic_bins": bins,
        "class_train_ids": class_ids,
        "budget_wire_bytes": budget_bytes,
        "methods": list(METHODS), "seed": seed,
        "channel": "synthetic_first_pilot_gilbert_elliott_entropy_packets_not_5g",
        "scenario": SCENARIO, "level": LEVEL,
        "target_packet_loss_rate": float(_PROFILES[SCENARIO]["loss_rates"][LEVEL]),
        "mean_burst_packets": MEAN_BURST_PACKETS,
        "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
        "packet_header_bytes": PACKET_HEADER_BYTES,
        "loss_fairness": "same deterministic GE event prefix per image/seed/packet ordinal",
        "jpeg_restart_markers": False, "jpeg_subsampling": int(config.section("image")["jpeg_subsampling"]),
        "jpeg_structure_delivery": "reliable and counted; only entropy packets lossy",
        "limitnet_control_delivery": "reliable and counted; model latent packets lossy",
        "limitnet_parity_packets_per_layer": 0,
        "limitnet_mode": limitnet_mode,
        "limitnet_model_variant": variant,
        "limitnet_checkpoint": str(checkpoint),
        "limitnet_checkpoint_sha256": _sha256_file(checkpoint),
        "limitnet_checkpoint_source": metadata.get("source"),
        "limitnet_training_dataset": metadata.get("training_dataset"),
        "limitnet_codec_input_resolution": ([224, 224] if limitnet_mode == "official224"
                                               else [2048, 1024]),
        "receiver_checkpoint": str(receiver_path),
        "receiver_checkpoint_sha256": _sha256_file(receiver_path),
        "receiver_missing_mask_policy": "all_zero_unlocalized_no_rst",
        "segmenter_revision": config.section("models")["segmenter"]["revision"],
        "segmenter_tile": int(config.section("image")["segformer_tile"]),
        "segmenter_stride": int(config.section("image")["segformer_stride"]),
        "kb_path": str(kb_path),
        "roi_weight": roi_weight, "background_weight": background_weight,
        "failure_policy": "infeasible/undecodable TP=FP=0 FN=all GT class pixels",
    }
    settings_sha256 = _settings_file(output / "settings.json", settings)
    segmenter = CityscapesSegmenter(config.model_path("segmenter"), device,
                                     settings["segmenter_tile"], settings["segmenter_stride"])
    shared = SharedKB.from_config(config, device)
    kb = PosteriorKB.load(kb_path)
    receiver = load_receiver(config, device)
    limitnet = load_limitnet(checkpoint, device)
    done = 0
    for position in range(shard_index, max_images, num_shards):
        image_path, label_path = pairs[position]
        image_id = image_path.stem
        case_dir = output / "cases" / f"{position + 1:04d}_{image_id}"
        paths = {method: _method_path(case_dir, method) for method in METHODS}
        if all(_stored(path, settings_sha256=settings_sha256, image_id=image_id,
                       method=method) is not None for method, path in paths.items()):
            done += 1
            continue
        _, current_bins, stage_folder = _wait_for_sample(
            config, stage12, position, image_path,
            time.monotonic() + wait_stage12_seconds)
        if current_bins != bins:
            raise ValueError("运行中语义桶改变，拒绝混合结果")
        image, truth = load_cityscapes_pair(image_path, label_path)
        patch = int(config.section("image")["patch_size"])
        expected = (math.ceil(image.height / patch), math.ceil(image.width / patch))
        alpha = np.load(stage_folder / "patch_alpha_fp32.npy", allow_pickle=False)
        selected = np.load(stage_folder / "selected_patch_mask.npy", allow_pickle=False)
        if (alpha.shape != expected or selected.shape != expected or
                selected.dtype != np.bool_ or not np.isfinite(alpha).all() or
                np.any(alpha < 0)):
            raise ValueError(f"阶段二重要性缓存形状或数值无效：{stage_folder}")
        case_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for method in METHODS:
            if _stored(paths[method], settings_sha256=settings_sha256,
                       image_id=image_id, method=method) is not None:
                continue
            save = (case_dir / f"{method}.png" if position < save_examples else None)
            if method == "kdc_restored":
                row = _kdc_case(
                    config, image, truth, image_id, bins, alpha, selected,
                    budget_bytes=budget_bytes, seed=seed,
                    roi_weight=roi_weight, background_weight=background_weight,
                    receiver=(None, shared, kb, receiver, device), segmenter=segmenter,
                    class_ids=class_ids, save_image=save)
            else:
                row = _limit_case(
                    image, truth, image_id, budget_bytes=budget_bytes, seed=seed,
                    codec=limitnet, mode=limitnet_mode, segmenter=segmenter,
                    class_ids=class_ids, save_image=save)
            row.update({"image_id": image_id, "image_index": position + 1,
                        "source_image": str(image_path), "method": method,
                        "settings_sha256": settings_sha256,
                        "elapsed_seconds_for_image_so_far": time.perf_counter() - started})
            _atomic_json(paths[method], row)
        image.close()
        done += 1
        print(f"[Fig8a GE shard {shard_index} {done}/{math.ceil(max_images / num_shards)}] "
              f"{image_id} {time.perf_counter() - started:.1f}s", flush=True)
    return {"output": str(output), "shard_index": shard_index,
            "num_shards": num_shards, "max_images": max_images,
            "completed_by_this_shard": done, "settings_sha256": settings_sha256}


def _pooled_iou(counts: np.ndarray) -> float | None:
    total = counts.sum(axis=0)
    union = int(total.sum())
    return float(total[0] / union) if union else None


def summarize_random_val(output: str | Path, *, max_images: int,
                         bootstrap: int = 2000, seed: int = 2026) -> dict:
    """合并并行分片；失败图作为零预测计入所有类别的分母。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output)
    settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
    if settings.get("protocol") != "figure8a_single_jpeg_synthetic_ge_val_v1":
        raise ValueError("不是最新单 JPEG GE 图 8(a) 结果目录")
    if not 1 <= max_images <= 500 or bootstrap < 100:
        raise ValueError("样本数或 bootstrap 次数无效")
    settings_sha256 = sha256(_json_bytes(settings)).hexdigest()
    bins = settings["semantic_bins"]
    case_dirs = sorted((output / "cases").glob("[0-9][0-9][0-9][0-9]_*"))
    rows: list[dict[str, dict]] = []
    for position in range(1, max_images + 1):
        matching = [folder for folder in case_dirs if folder.name.startswith(f"{position:04d}_")]
        if len(matching) != 1:
            raise ValueError(f"缺少或重复第 {position} 张 val 逐图结果")
        by_method = {}
        for method in METHODS:
            path = _method_path(matching[0], method)
            row = _stored(path, settings_sha256=settings_sha256,
                          image_id=matching[0].name.split("_", 1)[1], method=method)
            if row is None or set(row["counts"]) != set(bins):
                raise ValueError(f"第 {position} 张 {method} 不完整")
            by_method[method] = row
        rows.append(by_method)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, max_images, size=(bootstrap, max_images), dtype=np.int32)
    summary: dict[str, dict] = {}
    for method in METHODS:
        method_rows = [pair[method] for pair in rows]
        method_summary: dict = {
            "n_images": max_images,
            "n_ok": sum(row["status"] == "ok" for row in method_rows),
            "n_undecodable": sum(row["status"] == "undecodable" for row in method_rows),
            "n_infeasible": sum(row["status"] == "infeasible" for row in method_rows),
            "mean_sent_bytes": float(np.mean([row["sent_bytes"] for row in method_rows
                                               if row["sent_bytes"] is not None]))
                               if any(row["sent_bytes"] is not None for row in method_rows) else None,
            "mean_lost_packets": float(np.mean([row["lost_packets"] for row in method_rows])),
            "classes": {},
        }
        for name in bins:
            counts = np.asarray([row["counts"][name] for row in method_rows], dtype=np.int64)
            observed = _pooled_iou(counts)
            sampled_counts = counts[sampled].sum(axis=1)
            union = sampled_counts.sum(axis=1)
            distribution = np.divide(sampled_counts[:, 0], union,
                                     out=np.full(bootstrap, np.nan), where=union > 0)
            finite = distribution[np.isfinite(distribution)]
            ci = ([float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
                  if finite.size else None)
            method_summary["classes"][name] = {
                "iou": observed, "bootstrap_95_ci": ci,
                "counts_tp_fp_fn": counts.sum(axis=0).tolist(),
                "bootstrap_valid_draws": int(finite.size),
            }
        summary[method] = method_summary
    paired: dict[str, dict] = {}
    for name in bins:
        kdc = np.asarray([row["kdc_restored"]["counts"][name] for row in rows], dtype=np.int64)
        limit = np.asarray([row["limitnet"]["counts"][name] for row in rows], dtype=np.int64)
        kdc_draw = kdc[sampled].sum(axis=1)
        limit_draw = limit[sampled].sum(axis=1)
        kdc_union = kdc_draw.sum(axis=1)
        limit_union = limit_draw.sum(axis=1)
        valid = (kdc_union > 0) & (limit_union > 0)
        delta = kdc_draw[valid, 0] / kdc_union[valid] - limit_draw[valid, 0] / limit_union[valid]
        kdc_iou = summary["kdc_restored"]["classes"][name]["iou"]
        limit_iou = summary["limitnet"]["classes"][name]["iou"]
        paired[name] = {
            "kdc_minus_limitnet": (kdc_iou - limit_iou if kdc_iou is not None and
                                   limit_iou is not None else None),
            "bootstrap_95_ci": ([float(np.quantile(delta, 0.025)),
                                 float(np.quantile(delta, 0.975))]
                                if delta.size else None),
        }
    report = {"settings": settings, "evaluation_image_count": max_images,
              "full_validation": max_images == 500,
              "summary": summary, "paired_differences": paired,
              "failure_counting": settings["failure_policy"]}
    _atomic_json(output / f"report_n{max_images}.json", report)
    table = output / f"class_iou_n{max_images}.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "cityscapes_train_id", "method", "iou", "ci95_low",
                         "ci95_high", "tp", "fp", "fn", "n_ok", "n_undecodable",
                         "n_infeasible", "mean_sent_bytes"))
        for name in bins:
            for method in METHODS:
                item = summary[method]["classes"][name]
                method_summary = summary[method]
                interval = item["bootstrap_95_ci"] or (None, None)
                writer.writerow((name, settings["class_train_ids"][name], method,
                                 item["iou"], *interval,
                                 *item["counts_tp_fp_fn"],
                                 method_summary["n_ok"],
                                 method_summary["n_undecodable"],
                                 method_summary["n_infeasible"],
                                 method_summary["mean_sent_bytes"]))
    fig, axis = plt.subplots(figsize=(max(9.5, len(bins) * 1.6), 6.0))
    x = np.arange(len(bins), dtype=float)
    width = 0.36
    for offset, method, label, color in ((-0.5, "limitnet", "LimitNet", "#547da5"),
                                         (0.5, "kdc_restored", "KDC restored", "#c9323c")):
        entries = [summary[method]["classes"][name] for name in bins]
        values = np.array([item["iou"] if item["iou"] is not None else 0.0
                           for item in entries])
        lower = np.array([max(0, value - item["bootstrap_95_ci"][0])
                          if item["bootstrap_95_ci"] is not None else 0.0
                          for value, item in zip(values, entries)])
        upper = np.array([max(0, item["bootstrap_95_ci"][1] - value)
                          if item["bootstrap_95_ci"] is not None else 0.0
                          for value, item in zip(values, entries)])
        axis.bar(x + offset * width, values, width, color=color, label=label,
                 yerr=np.stack([lower, upper]), capsize=3)
        for position, item in enumerate(entries):
            if item["iou"] is None:
                axis.text(x[position] + offset * width, 0.025, "n/a",
                          rotation=90, ha="center", va="bottom", fontsize=9)
    axis.set_xticks(x, bins, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Class IoU")
    axis.set_title(f"Cityscapes val {max_images}/500 · 30 kB wire budget · synthetic GE loss"
                   if settings["budget_wire_bytes"] == 30_000 else
                   f"Cityscapes val {max_images}/500 · {settings['budget_wire_bytes']/1000:g} kB wire budget · synthetic GE loss")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.text(0.01, 0.015,
             f"LimitNet: {settings['limitnet_mode']} ({settings['limitnet_checkpoint_source']}); "
             "infeasible/undecodable images count as zero prediction. "
             "Paired image bootstrap 95% CI. No 5G trace or JPEG RST.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figure = output / f"class_iou_n{max_images}.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"report": str(output / f"report_n{max_images}.json"),
            "table": str(table), "figure": str(figure), "n_images": max_images,
            "summary": summary, "paired_differences": paired}
