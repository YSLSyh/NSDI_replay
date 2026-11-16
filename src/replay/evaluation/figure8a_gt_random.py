"""论文图 8(a) 的 Cityscapes 真值区域三任务对照。

每个目标类别分别从 val 人工标注生成重点 patch、重新编码一张图像。
传输仍沿用本项目当前约定：单 JPEG、无 RST、合成 GE 随机丢包。
真值只在本 oracle 实验的发送端构造重要性图，绝不能作为普通推理的输入。
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from replay.core.assets import verify_model_lock
from replay.core.config import ReplayConfig
from replay.data.cityscapes import cityscapes_pairs, load_cityscapes_pair
from replay.evaluation.figure8 import ground_truth_patches
from replay.evaluation.figure8a_packet import (
    LIMITNET_MODES, _validate_limitnet_checkpoint_metadata,
)
from replay.evaluation.figure8a_random import (
    LEVEL, MEAN_BURST_PACKETS, METHODS, PACKET_HEADER_BYTES,
    PACKET_PAYLOAD_BYTES, SCENARIO, _atomic_json, _counts,
    _decode_intact, _failure_counts, _is_budget_error, _json_bytes,
    _limit_case, _method_path, _pooled_iou, _psnr, _settings_file,
    _sha256_file, _stored,
)
from replay.transmission.channel import _PROFILES
from replay.transmission.header import FrameHeader, SessionHeader
from replay.transmission.plain_jpeg_channel import simulate_plain_jpeg_entropy_loss
from replay.transmission.task_jpeg_search import SearchProfile, search_task_jpeg


TARGETS = {"car": 13, "pedestrian": 11, "vegetation": 8}
FOCUS_SOURCE = "Cityscapes val ground_truth_labelIds oracle"
TRAIN_PROFILE_PROTOCOL = "figure8a_gt_train_only_30kb_prequant_v1"


def _bin_name(task: str) -> str:
    return "person" if task == "pedestrian" else task


def _load_sender_profiles(
    config: ReplayConfig, source: str | Path, *, budget_bytes: int,
    val_image_ids: set[str], roi_weight: float, background_weight: float,
) -> tuple[dict[str, tuple[SearchProfile, int]], dict]:
    """只接受 train 冻结的三类参数，拒绝在 val 上临时选量化强度。"""
    if budget_bytes != 30_000:
        raise ValueError("图 8(a) 冻结发送端要求统一 30,000 字节传输预算")
    source_path = config.path(source)
    raw = source_path.read_bytes()
    report = json.loads(raw)
    settings = report.get("settings", {})
    if (settings.get("protocol") != TRAIN_PROFILE_PROTOCOL or
            settings.get("split") != "train" or
            settings.get("budget_wire_bytes") != budget_bytes or
            settings.get("native_resolution") != [2048, 1024] or
            settings.get("patch_size") != int(config.section("image")["patch_size"]) or
            settings.get("jpeg_restart_markers") is not False or
            settings.get("background_blur") is not False or
            settings.get("packet_payload_bytes") != PACKET_PAYLOAD_BYTES or
            settings.get("packet_header_bytes") != PACKET_HEADER_BYTES or
            settings.get("roi_weight") != roi_weight or
            settings.get("background_weight") != background_weight or
            report.get("all_train_samples_feasible") is not True):
        raise ValueError("发送端参数并非同配置、同预算的 train 可行冻结结果")
    config_sha256 = sha256(json.dumps(config.values, sort_keys=True,
                                     default=str).encode()).hexdigest()
    if settings.get("config_sha256") != config_sha256:
        raise ValueError("发送端标定与验证使用的项目配置不同")
    samples = settings.get("sample_ids_by_task")
    selected = report.get("selected_by_task")
    if (not isinstance(samples, dict) or set(samples) != set(TARGETS) or
            not isinstance(selected, dict) or set(selected) != set(TARGETS)):
        raise ValueError("冻结结果缺少三个独立类别")
    result: dict[str, tuple[SearchProfile, int]] = {}
    metadata: dict[str, dict] = {}
    for task in TARGETS:
        ids = samples[task]
        if (not isinstance(ids, list) or not ids or
                not all(isinstance(item, str) and item for item in ids) or
                len(set(ids)) != len(ids) or not set(ids).isdisjoint(val_image_ids)):
            raise ValueError(f"{task} 的训练选样为空、重复或混入 val")
        entry = selected[task]
        if not isinstance(entry, dict) or entry.get("train_feasible_rate") != 1.0:
            raise ValueError(f"{task} 在 train 标定样本上尚未全部满足 30 kB")
        try:
            profile = SearchProfile(**entry["profile"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"{task} 的预量化参数无效") from error
        if (not profile.protected_selected or
                (profile.strength <= 0 and
                 profile.background_low_frequency_strength <= 0) or
                not all(math.isfinite(value) and value >= 0 for value in
                        (profile.strength, profile.zero_threshold_multiplier,
                         profile.background_low_frequency_strength)) or
                profile.cutoff != int(config.section("importance")["dct_high_frequency_cutoff"])):
            raise ValueError(f"{task} 缺少有效的重点区域保护和背景预量化")
        subsampling = entry.get("subsampling")
        if type(subsampling) is not int or subsampling not in (0, 2):
            raise ValueError(f"{task} JPEG 色度采样设置无效")
        result[task] = profile, subsampling
        metadata[task] = {
            "candidate": entry.get("candidate"),
            "profile": asdict(profile),
            "subsampling": subsampling,
            "train_feasible_rate": entry["train_feasible_rate"],
            "train_sample_ids": ids,
        }
    return result, {
        "source": "Cityscapes train frozen sender tuning",
        "path": str(source_path),
        "sha256": sha256(raw).hexdigest(),
        "protocol": TRAIN_PROFILE_PROTOCOL,
        "selected_by_task": metadata,
    }


def _oracle_kdc_case(
    config: ReplayConfig, image: Image.Image, truth: np.ndarray,
    image_id: str, task: str, *, first_in_session: bool,
    budget_bytes: int, seed: int, roi_weight: float,
    background_weight: float, receiver: tuple, segmenter,
    save_image: Path | None, sender_profile: SearchProfile | None = None,
    sender_subsampling: int = 0,
) -> dict:
    """使用验证集真值的布尔 patch 图；目标 patch 不受背景粗量化。"""
    from replay.receiver.infer import restore

    class_id = TARGETS[task]
    patch = int(config.section("image")["patch_size"])
    selected = ground_truth_patches(truth, class_id, patch)
    target_pixels = int(np.count_nonzero(truth == class_id))
    clip = config.section("models")["clip"]
    bins = [_bin_name(task)]
    session = SessionHeader(task, tuple(bins), clip["repo"], clip["revision"]).to_bytes()
    frame = FrameHeader(image.width, image.height, patch, selected).to_bytes()
    charged_session_bytes = len(session) if first_in_session else 0
    control_bytes = charged_session_bytes + len(frame)
    base = {
        "method": "kdc_restored", "task": task, "status": "infeasible",
        "sent_bytes": None, "packet_count": 0, "lost_packets": 0,
        "lost_packet_indices": [], "receiver_applied": False,
        "semantic_control_bytes": control_bytes,
        "session_header_bytes_charged": charged_session_bytes,
        "frame_header_bytes": len(frame),
        "focus_source": FOCUS_SOURCE,
        "selection_uses_validation_labels": True,
        "gt_target_pixels": target_pixels,
        "gt_target_present": bool(target_pixels),
        "gt_selected_patch_count": int(selected.sum()),
        "no_target_policy": "encode with all-zero focus; count any predicted false positives",
        "counts": _failure_counts(truth, {task: class_id}),
    }
    try:
        sender_options = ({"profiles": (sender_profile,),
                           "require_plain_fallback": False}
                          if sender_profile is not None else {})
        payload, search = search_task_jpeg(
            image, selected, budget_bytes=budget_bytes, patch_size=patch,
            selected_patch_mask=selected, roi_weight=roi_weight,
            background_weight=background_weight,
            packet_payload_bytes=PACKET_PAYLOAD_BYTES,
            packet_header_bytes=PACKET_HEADER_BYTES,
            fixed_control_bytes=control_bytes,
            max_quality=95,
            subsampling=sender_subsampling,
            cutoff=int(config.section("importance")["dct_high_frequency_cutoff"]),
            gamma=float(config.section("importance")["prequant_gamma"]),
            importance_ratio_low=float(config.section("importance")["importance_ratio_low"]),
            importance_ratio_high=float(config.section("importance")["importance_ratio_high"]),
            **sender_options,
        )
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
        mean_burst_packets=MEAN_BURST_PACKETS,
    )
    total_sent = outcome.sent_bytes + control_bytes
    if total_sent != search["codec_wire_bytes"] or total_sent > budget_bytes:
        raise AssertionError("GT KDC 搜索字节账本与实际传输不一致")
    # search_task_jpeg 的通用默认值为 False；本调用的输入实际来自 val 真值。
    search = dict(search)
    search["selection_uses_validation_labels"] = True
    search["focus_source"] = FOCUS_SOURCE
    if sender_profile is not None and search["plain_jpeg"]:
        raise AssertionError("冻结的 KDC 参数意外退化为普通 JPEG")
    fields = {
        **base, "status": outcome.status, "sent_bytes": total_sent,
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
        "search_selection": search,
    }
    if outcome.decoded is None:
        return fields
    if outcome.lost_packets:
        shared, kb, model, device = receiver
        restored = restore(
            outcome.decoded, bins, shared, kb, model,
            int(config.section("knowledge")["rx_top_k"]), device,
            update_session=False, missing_pixels=None,
        )
        fields["receiver_applied"] = True
    else:
        restored = outcome.decoded
    if save_image is not None:
        restored.save(save_image)
    fields["counts"] = _counts(segmenter.predict(restored), truth, {task: class_id})
    fields["psnr_db"] = _psnr(image, restored)
    return fields


def _settings(
    config: ReplayConfig, pairs: list[tuple[Path, Path]], task: str,
    *, checkpoint: Path, metadata: dict, variant: str,
    limitnet_mode: str, budget_bytes: int, seed: int,
    roi_weight: float, background_weight: float,
    sender_profile_metadata: dict | None,
    receiver_checkpoint_override: str | Path | None = None,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    receiver_path = (config.receiver_checkpoint() if receiver_checkpoint_override is None
                     else config.path(receiver_checkpoint_override))
    return {
        "protocol": "figure8a_gt_oracle_single_jpeg_synthetic_ge_val_v1",
        "task": task, "cityscapes_train_id": TARGETS[task],
        "split": "val", "n_available_val_images": 500,
        "val_image_ids_sha256": sha256("\n".join(path.stem for path, _ in pairs).encode()).hexdigest(),
        "focus_source": FOCUS_SOURCE,
        "selection_uses_validation_labels": True,
        "focus_patch_rule": "any target trainId pixel in 16x16 patch",
        "target_semantic_bin": _bin_name(task),
        "one_encoding_per_target_task": True,
        "no_target_policy": "encode with all-zero focus; count any predicted false positives",
        "session_header_charge_rule": "once at the first val image of each target task",
        "budget_wire_bytes": budget_bytes,
        "methods": list(methods), "seed": seed,
        "channel": "synthetic_first_pilot_gilbert_elliott_entropy_packets_not_5g",
        "scenario": SCENARIO, "level": LEVEL,
        "target_packet_loss_rate": float(_PROFILES[SCENARIO]["loss_rates"][LEVEL]),
        "mean_burst_packets": MEAN_BURST_PACKETS,
        "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
        "packet_header_bytes": PACKET_HEADER_BYTES,
        "loss_fairness": "same deterministic GE event prefix per image/seed/packet ordinal",
        "jpeg_restart_markers": False,
        "jpeg_subsampling": (sender_profile_metadata["selected_by_task"][task]["subsampling"]
                             if sender_profile_metadata is not None else
                             int(config.section("image")["jpeg_subsampling"])),
        "sender_profile_source": (sender_profile_metadata["source"]
                                  if sender_profile_metadata is not None else
                                  "unfrozen_default_smoke_only"),
        "sender_profile_config_sha256": (sender_profile_metadata["sha256"]
                                         if sender_profile_metadata is not None else None),
        "sender_profile_selected": (sender_profile_metadata["selected_by_task"][task]
                                    if sender_profile_metadata is not None else None),
        "jpeg_structure_delivery": "reliable and counted; only entropy packets lossy",
        "limitnet_control_delivery": "reliable and counted; model latent packets lossy",
        "limitnet_parity_packets_per_layer": 0,
        "limitnet_mode": limitnet_mode,
        "limitnet_model_variant": variant,
        "limitnet_checkpoint": str(checkpoint),
        "limitnet_checkpoint_sha256": _sha256_file(checkpoint),
        "limitnet_checkpoint_source": metadata.get("source"),
        "limitnet_training_dataset": metadata.get("training_dataset"),
        "limitnet_target_class_ids": metadata.get("target_class_ids"),
        "limitnet_codec_input_resolution": ([224, 224] if limitnet_mode == "official224"
                                               else [2048, 1024]),
        "receiver_checkpoint": str(receiver_path),
        "receiver_checkpoint_sha256": _sha256_file(receiver_path),
        "receiver_missing_mask_policy": "all_zero_unlocalized_no_rst",
        "segmenter_revision": config.section("models")["segmenter"]["revision"],
        "segmenter_tile": int(config.section("image")["segformer_tile"]),
        "segmenter_stride": int(config.section("image")["segformer_stride"]),
        "kb_path": str(config.data_path("kb_path")),
        "roi_weight": roi_weight, "background_weight": background_weight,
        "failure_policy": "infeasible/undecodable TP=FP=0 FN=all GT target pixels",
        "paper_difference": "synthetic GE rather than measured 5G trace; unpublished JPEG quantization details are project choices",
    }


def run_gt_val(
    config: ReplayConfig, *, output: str | Path,
    limitnet_checkpoint: str | Path, task: str = "all",
    limitnet_mode: str = "native_adapted", device: torch.device,
    budget_bytes: int = 30_000, max_images: int = 100,
    seed: int = 2026, num_shards: int = 1, shard_index: int = 0,
    roi_weight: float = 4.0, background_weight: float = 1.0,
    save_examples: int = 0,
    sender_profile_config: str | Path | None = None,
    _sender_profile_loader=None,
    _kdc_case_runner=None,
    _settings_builder=None,
    _extra_method_runner=None,
    _methods: tuple[str, ...] = METHODS,
    _receiver_checkpoint_override: str | Path | None = None,
    _pilot_pair_selector=None,
) -> dict:
    """默认按官方 val 固定前缀评测；RST pilot 可显式传入跨城市抽样器。"""
    from replay.baselines.limitnet import load_limitnet
    from replay.evaluation.segmentation import CityscapesSegmenter
    from replay.knowledge.encoders import SharedKB
    from replay.knowledge.store import PosteriorKB
    from replay.receiver.infer import load_receiver

    if (task not in (*TARGETS, "all") or limitnet_mode not in LIMITNET_MODES or
            budget_bytes <= 0 or not 1 <= max_images <= 500 or
            num_shards < 1 or not 0 <= shard_index < num_shards or
            save_examples < 0 or roi_weight <= 0 or background_weight <= 0 or
            "kdc_restored" not in _methods or "limitnet" not in _methods or
            ("jpeg_rst" in _methods) != (_extra_method_runner is not None) or
            any(method not in ("kdc_restored", "limitnet", "jpeg_rst") for method in _methods)):
        raise ValueError("GT 图 8(a) 评测参数无效")
    if sender_profile_config is None and max_images > 2:
        raise ValueError("正式 GT 验证必须提供 Cityscapes train 冻结的 --sender-profile-config")
    pairs = list(cityscapes_pairs(config.data_path("cityscapes_root"), "val"))
    if len(pairs) != 500:
        raise ValueError(f"Cityscapes val 应有 500 张完整标注图，实际 {len(pairs)}")
    if _pilot_pair_selector is None:
        evaluated_pairs = pairs
        pilot_selection = None
    else:
        evaluated_pairs, pilot_selection = _pilot_pair_selector(pairs, max_images, seed)
        if (len(evaluated_pairs) != max_images or
                len({image.stem for image, _ in evaluated_pairs}) != max_images or
                not {image.stem for image, _ in evaluated_pairs}.issubset(
                    {image.stem for image, _ in pairs})):
            raise ValueError("pilot 抽样器没有返回 val 中互不重复的指定数量图像")
    sender_profiles, sender_metadata = (
        (_sender_profile_loader or _load_sender_profiles)(
            config, sender_profile_config, budget_bytes=budget_bytes,
            val_image_ids={path.stem for path, _ in pairs},
            roi_weight=roi_weight, background_weight=background_weight,
        ) if sender_profile_config is not None else ({}, None))
    output = config.path(output)
    checkpoint = config.path(limitnet_checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    variant = _validate_limitnet_checkpoint_metadata(limitnet_mode, metadata)
    if metadata.get("target_class_ids") is not None and set(metadata["target_class_ids"]) != set(TARGETS.values()):
        raise ValueError("LimitNet checkpoint 的任务类别不是 car/pedestrian/vegetation")
    del payload
    verify_model_lock(config, "segmenter")
    segmenter = CityscapesSegmenter(
        config.model_path("segmenter"), device,
        int(config.section("image")["segformer_tile"]),
        int(config.section("image")["segformer_stride"]),
    )
    shared = SharedKB.from_config(config, device)
    kb = PosteriorKB.load(config.data_path("kb_path"))
    receiver = load_receiver(config, device,
                             checkpoint_path=_receiver_checkpoint_override)
    limitnet = load_limitnet(checkpoint, device)
    tasks = tuple(TARGETS) if task == "all" else (task,)
    task_dirs: dict[str, Path] = {}
    task_hashes: dict[str, str] = {}
    for target in tasks:
        target_dir = output / target
        settings = (_settings_builder or _settings)(
            config, pairs, target, checkpoint=checkpoint,
            metadata=metadata, variant=variant, limitnet_mode=limitnet_mode,
            budget_bytes=budget_bytes, seed=seed,
            roi_weight=roi_weight, background_weight=background_weight,
            sender_profile_metadata=sender_metadata,
            receiver_checkpoint_override=_receiver_checkpoint_override,
            methods=_methods,
        )
        if pilot_selection is not None:
            settings["pilot_selection"] = pilot_selection
        task_dirs[target] = target_dir
        task_hashes[target] = _settings_file(target_dir / "settings.json", settings)

    completed = {target: 0 for target in tasks}
    shard_total = len(range(shard_index, max_images, num_shards))
    for position in range(shard_index, max_images, num_shards):
        image_path, label_path = evaluated_pairs[position]
        image_id = image_path.stem
        case_dirs = {
            target: task_dirs[target] / "cases" / f"{position + 1:04d}_{image_id}"
            for target in tasks
        }
        paths = {
            target: {method: _method_path(case_dirs[target], method)
                     for method in _methods}
            for target in tasks
        }
        missing = {
            target: {
                method for method in _methods
                if _stored(paths[target][method], settings_sha256=task_hashes[target],
                           image_id=image_id, method=method) is None
            }
            for target in tasks
        }
        if not any(missing.values()):
            for target in tasks:
                completed[target] += 1
            continue

        image, truth = load_cityscapes_pair(image_path, label_path)
        started = time.perf_counter()
        try:
            # LimitNet 与目标语义桶无关，同一原图只编码、传输、分割一次。
            # 各任务目录仍保留独立 JSON，便于与已有 2 图结果续接。
            needs_limitnet = [target for target in tasks if "limitnet" in missing[target]]
            if needs_limitnet:
                first = needs_limitnet[0]
                for target in needs_limitnet:
                    case_dirs[target].mkdir(parents=True, exist_ok=True)
                first_image = (case_dirs[first] / "limitnet.png"
                               if position < save_examples else None)
                shared_row = _limit_case(
                    image, truth, image_id, budget_bytes=budget_bytes,
                    seed=seed, codec=limitnet, mode=limitnet_mode,
                    segmenter=segmenter,
                    class_ids={target: TARGETS[target] for target in tasks},
                    save_image=first_image,
                )
                for target in needs_limitnet:
                    if first_image is not None and first_image.is_file() and target != first:
                        shutil.copyfile(first_image, case_dirs[target] / "limitnet.png")
                    row = {**shared_row, "counts": {target: shared_row["counts"][target]}}
                    row.update({
                        "task": target, "image_id": image_id,
                        "image_index": position + 1,
                        "source_image": str(image_path), "method": "limitnet",
                        "settings_sha256": task_hashes[target],
                        "elapsed_seconds_for_image_so_far": time.perf_counter() - started,
                    })
                    _atomic_json(paths[target]["limitnet"], row)

            for target in tasks:
                if _extra_method_runner is not None and "jpeg_rst" in missing[target]:
                    case_dirs[target].mkdir(parents=True, exist_ok=True)
                    extra_save = (case_dirs[target] / "jpeg_rst.png"
                                  if position < save_examples else None)
                    extra = _extra_method_runner(
                        config, image, truth, image_id, target,
                        first_in_session=position == 0,
                        budget_bytes=budget_bytes, seed=seed,
                        roi_weight=roi_weight, background_weight=background_weight,
                        receiver=(shared, kb, receiver, device), segmenter=segmenter,
                        save_image=extra_save,
                        sender_profile_entry=(sender_profiles[target]
                                              if sender_metadata is not None else None),
                    )
                    extra.update({
                        "task": target, "image_id": image_id,
                        "image_index": position + 1,
                        "source_image": str(image_path), "method": "jpeg_rst",
                        "settings_sha256": task_hashes[target],
                        "elapsed_seconds_for_image_so_far": time.perf_counter() - started,
                    })
                    _atomic_json(paths[target]["jpeg_rst"], extra)
                if "kdc_restored" not in missing[target]:
                    continue
                case_dirs[target].mkdir(parents=True, exist_ok=True)
                save = (case_dirs[target] / "kdc_restored.png"
                        if position < save_examples else None)
                kdc_runner = _kdc_case_runner or _oracle_kdc_case
                kdc_kwargs = ({"sender_profile_entry": sender_profiles[target]}
                              if _kdc_case_runner is not None and sender_metadata is not None
                              else {})
                row = kdc_runner(
                    config, image, truth, image_id, target,
                    first_in_session=position == 0,
                    budget_bytes=budget_bytes, seed=seed,
                    roi_weight=roi_weight, background_weight=background_weight,
                    receiver=(shared, kb, receiver, device), segmenter=segmenter,
                    save_image=save,
                    sender_profile=(sender_profiles[target][0]
                                    if sender_metadata is not None else None),
                    sender_subsampling=(sender_profiles[target][1]
                                        if sender_metadata is not None else
                                        int(config.section("image")["jpeg_subsampling"])),
                    **kdc_kwargs,
                )
                row.update({
                    "task": target, "image_id": image_id,
                    "image_index": position + 1,
                    "source_image": str(image_path), "method": "kdc_restored",
                    "settings_sha256": task_hashes[target],
                    "elapsed_seconds_for_image_so_far": time.perf_counter() - started,
                })
                _atomic_json(paths[target]["kdc_restored"], row)
        finally:
            image.close()
        for target in tasks:
            completed[target] += 1
        print(f"[Fig8a GT {task} shard {shard_index} "
              f"{completed[tasks[0]]}/{shard_total}] "
              f"{image_id} {time.perf_counter() - started:.1f}s", flush=True)
    result = {
        "output": str(output), "tasks": list(tasks),
        "shard_index": shard_index, "num_shards": num_shards,
        "max_images": max_images, "completed_by_task": completed,
    }
    if pilot_selection is not None:
        result["pilot_selection"] = pilot_selection
    return result


def summarize_gt_val(output: str | Path, *, max_images: int,
                     bootstrap: int = 2000, seed: int = 2026,
                     expected_protocol: str = "figure8a_gt_oracle_single_jpeg_synthetic_ge_val_v1",
                     figure_note: str | None = None) -> dict:
    """三任务独立重编码的结果汇总；失败样本保留在 IoU 分母。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output)
    if not 1 <= max_images <= 500 or bootstrap < 100:
        raise ValueError("样本数或 bootstrap 次数无效")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, max_images, size=(bootstrap, max_images), dtype=np.int32)
    summary: dict[str, dict] = {}
    paired: dict[str, dict] = {}
    settings_by_task: dict[str, dict] = {}
    methods: tuple[str, ...] | None = None
    pilot_selection: dict | None = None
    pilot_mode_seen: bool | None = None
    for target in TARGETS:
        target_dir = output / target
        settings = json.loads((target_dir / "settings.json").read_text(encoding="utf-8"))
        if (settings.get("protocol") != expected_protocol or
                settings.get("task") != target or
                settings.get("focus_source") != FOCUS_SOURCE):
            raise ValueError(f"{target} 不是本次真值区域实验")
        selected = settings.get("pilot_selection")
        if pilot_mode_seen is not None and pilot_mode_seen != (selected is not None):
            raise ValueError("三个类别混用了默认前缀和跨城市 pilot 样本")
        pilot_mode_seen = selected is not None
        if selected is not None:
            ids = selected.get("selected_image_ids", [])
            if (max_images == 500 or not isinstance(ids, list) or
                    len(ids) != max_images or len(set(ids)) != max_images or
                    selected.get("selected_image_ids_sha256") !=
                    sha256("\n".join(ids).encode()).hexdigest() or
                    (pilot_selection is not None and selected != pilot_selection)):
                raise ValueError("跨城市 pilot 样本清单、哈希或类别间顺序不一致")
            pilot_selection = selected
        task_methods = tuple(settings.get("methods", ()))
        if ("kdc_restored" not in task_methods or "limitnet" not in task_methods or
                not set(task_methods).issubset({"kdc_restored", "limitnet", "jpeg_rst"}) or
                len(task_methods) != len(set(task_methods)) or
                (methods is not None and methods != task_methods)):
            raise ValueError("三个类别的对照方法必须一致")
        methods = task_methods
        settings_by_task[target] = settings
        settings_sha256 = sha256(_json_bytes(settings)).hexdigest()
        case_dirs = sorted((target_dir / "cases").glob("[0-9][0-9][0-9][0-9]_*"))
        rows: list[dict[str, dict]] = []
        for position in range(1, max_images + 1):
            matching = [folder for folder in case_dirs if folder.name.startswith(f"{position:04d}_")]
            if len(matching) != 1:
                raise ValueError(f"{target} 缺少或重复第 {position} 张 val 逐图结果")
            if (pilot_selection is not None and
                    matching[0].name.split("_", 1)[1] !=
                    pilot_selection["selected_image_ids"][position - 1]):
                raise ValueError(f"{target} 第 {position} 张与跨城市 pilot 清单不一致")
            by_method = {}
            for method in methods:
                row = _stored(
                    _method_path(matching[0], method),
                    settings_sha256=settings_sha256,
                    image_id=matching[0].name.split("_", 1)[1], method=method,
                )
                if row is None or set(row["counts"]) != {target}:
                    raise ValueError(f"{target} 第 {position} 张 {method} 不完整")
                by_method[method] = row
            rows.append(by_method)
        method_summary: dict[str, dict] = {}
        for method in methods:
            method_rows = [row[method] for row in rows]
            counts = np.asarray([row["counts"][target] for row in method_rows], dtype=np.int64)
            draws = counts[sampled].sum(axis=1)
            union = draws.sum(axis=1)
            distribution = np.divide(draws[:, 0], union,
                                     out=np.full(bootstrap, np.nan), where=union > 0)
            finite = distribution[np.isfinite(distribution)]
            method_summary[method] = {
                "iou": _pooled_iou(counts),
                "bootstrap_95_ci": ([float(np.quantile(finite, 0.025)),
                                     float(np.quantile(finite, 0.975))]
                                    if finite.size else None),
                "counts_tp_fp_fn": counts.sum(axis=0).tolist(),
                "n_images": max_images,
                "n_ok": sum(row["status"] == "ok" for row in method_rows),
                "n_undecodable": sum(row["status"] == "undecodable" for row in method_rows),
                "n_infeasible": sum(row["status"] == "infeasible" for row in method_rows),
                "mean_sent_bytes": (float(np.mean([row["sent_bytes"] for row in method_rows
                                                   if row["sent_bytes"] is not None]))
                                    if any(row["sent_bytes"] is not None for row in method_rows)
                                    else None),
                "gt_absent_images": sum(row.get("gt_target_present") is False
                                        for row in method_rows) if method == "kdc_restored" else None,
                "sender_used_importance_images": sum(row.get("sender_used_importance") is True
                                                     for row in method_rows) if method == "kdc_restored" else None,
            }
        summary[target] = method_summary
        kdc = np.asarray([row["kdc_restored"]["counts"][target] for row in rows], dtype=np.int64)
        limit = np.asarray([row["limitnet"]["counts"][target] for row in rows], dtype=np.int64)
        kdc_draw = kdc[sampled].sum(axis=1)
        limit_draw = limit[sampled].sum(axis=1)
        kdc_union = kdc_draw.sum(axis=1)
        limit_union = limit_draw.sum(axis=1)
        valid = (kdc_union > 0) & (limit_union > 0)
        delta = kdc_draw[valid, 0] / kdc_union[valid] - limit_draw[valid, 0] / limit_union[valid]
        kiou, liou = method_summary["kdc_restored"]["iou"], method_summary["limitnet"]["iou"]
        paired[target] = {
            "kdc_minus_limitnet": (kiou - liou if kiou is not None and liou is not None else None),
            "bootstrap_95_ci": ([float(np.quantile(delta, 0.025)),
                                 float(np.quantile(delta, 0.975))] if delta.size else None),
        }
        if "jpeg_rst" in methods:
            jpeg = np.asarray([row["jpeg_rst"]["counts"][target] for row in rows], dtype=np.int64)
            jpeg_draw = jpeg[sampled].sum(axis=1)
            jpeg_union = jpeg_draw.sum(axis=1)
            jpeg_valid = (kdc_union > 0) & (jpeg_union > 0)
            jpeg_delta = (kdc_draw[jpeg_valid, 0] / kdc_union[jpeg_valid] -
                          jpeg_draw[jpeg_valid, 0] / jpeg_union[jpeg_valid])
            jiou = method_summary["jpeg_rst"]["iou"]
            paired[target]["kdc_minus_jpeg_rst"] = (
                kiou - jiou if kiou is not None and jiou is not None else None)
            paired[target]["kdc_minus_jpeg_rst_bootstrap_95_ci"] = (
                [float(np.quantile(jpeg_delta, 0.025)),
                 float(np.quantile(jpeg_delta, 0.975))] if jpeg_delta.size else None)
    report = {
        "protocol": expected_protocol,
        "focus_source": FOCUS_SOURCE,
        "selection_uses_validation_labels": True,
        "evaluation_image_count_per_task": max_images,
        "full_validation": max_images == 500,
        "settings_by_task": settings_by_task,
        "summary": summary, "paired_differences": paired,
        "failure_counting": "infeasible/undecodable image predicts no target pixels",
    }
    if pilot_selection is not None:
        report["pilot_selection"] = pilot_selection
    _atomic_json(output / f"report_n{max_images}.json", report)
    table = output / f"class_iou_n{max_images}.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "cityscapes_train_id", "method", "iou", "ci95_low",
                         "ci95_high", "tp", "fp", "fn", "n_ok", "n_undecodable",
                         "n_infeasible", "mean_sent_bytes", "gt_absent_images",
                         "sender_used_importance_images"))
        for target, class_id in TARGETS.items():
            for method in methods:
                item = summary[target][method]
                interval = item["bootstrap_95_ci"] or (None, None)
                writer.writerow((target, class_id, method, item["iou"], *interval,
                                 *item["counts_tp_fp_fn"], item["n_ok"],
                                 item["n_undecodable"], item["n_infeasible"],
                                 item["mean_sent_bytes"], item["gt_absent_images"],
                                 item["sender_used_importance_images"]))
    fig, axis = plt.subplots(figsize=(9.5, 6.0))
    x = np.arange(len(TARGETS), dtype=float)
    # 保持旧版双柱图的 LimitNet 左、KDC 右版式，新增 RST 版按三方法顺序排。
    plot_methods = (("limitnet", "kdc_restored") if len(methods) == 2 else methods)
    width = 0.72 / len(plot_methods)
    palette = {"limitnet": ("LimitNet", "#547da5"),
               "kdc_restored": ("KDC restored", "#c9323c"),
               "jpeg_rst": ("JPEG + RST", "#8390a6")}
    for index, method in enumerate(plot_methods):
        label, color = palette[method]
        offset = index - (len(plot_methods) - 1) / 2
        entries = [summary[target][method] for target in TARGETS]
        values = np.asarray([entry["iou"] if entry["iou"] is not None else 0.0
                             for entry in entries])
        lower = np.asarray([max(0, value - entry["bootstrap_95_ci"][0])
                            if entry["bootstrap_95_ci"] is not None else 0.0
                            for value, entry in zip(values, entries)])
        upper = np.asarray([max(0, entry["bootstrap_95_ci"][1] - value)
                            if entry["bootstrap_95_ci"] is not None else 0.0
                            for value, entry in zip(values, entries)])
        axis.bar(x + offset * width, values, width, color=color, label=label,
                 yerr=np.stack((lower, upper)), capsize=3)
    axis.set_xticks(x, list(TARGETS))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Class IoU")
    budget = settings_by_task["car"]["budget_wire_bytes"]
    axis.set_title(f"Cityscapes GT oracle focus · {max_images}/500 val · "
                   f"{budget/1000:g} kB · synthetic GE loss")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.text(0.01, 0.015,
             figure_note or
             "Each category encoded separately from its validation GT mask. "
             "Failures count as zero prediction. No JPEG RST or measured 5G trace.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figure = output / f"class_iou_n{max_images}.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"report": str(output / f"report_n{max_images}.json"),
            "table": str(table), "figure": str(figure),
            "n_images_per_task": max_images, "summary": summary,
            "paired_differences": paired}
