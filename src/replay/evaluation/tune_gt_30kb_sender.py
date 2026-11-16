"""只用 Cityscapes train 为图 8(a) 选择 30 kB 发送端预量化参数。

这是论文未公开参数的工程标定。选样、搜索和排序只读取 train 图像与真值；
val 入口只能读取已冻结的参数文件，不得在 val 上重新选择强度。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image

from replay.core.config import ReplayConfig, write_json
from replay.data.cityscapes import cityscapes_pairs, load_cityscapes_pair
from replay.evaluation.figure8 import ground_truth_patches
from replay.transmission.header import FrameHeader, SessionHeader
from replay.transmission.task_jpeg_search import SearchProfile, search_task_jpeg


TARGETS = {"car": (13, 26), "pedestrian": (11, 24), "vegetation": (8, 21)}
PACKET_PAYLOAD_BYTES = 1200
PACKET_HEADER_BYTES = 8


@dataclass(frozen=True)
class Candidate:
    """一组标准 JPEG 参数；0=4:4:4，2=4:2:0。"""

    subsampling: int
    background_low_frequency_strength: float
    high_frequency_strength: float = 8.0
    zero_threshold_multiplier: float = 2.0
    cutoff: int = 2

    @property
    def name(self) -> str:
        chroma = "444" if self.subsampling == 0 else "420"
        return f"{chroma}_low{self.background_low_frequency_strength:g}"

    def profile(self) -> SearchProfile:
        return SearchProfile(
            strength=self.high_frequency_strength,
            zero_threshold_multiplier=self.zero_threshold_multiplier,
            protected_selected=True,
            cutoff=self.cutoff,
            background_low_frequency_strength=self.background_low_frequency_strength,
        )


def select_train_pairs(root: Path, *, positive_per_task: int,
                       negative_per_task: int, seed: int
                       ) -> dict[str, list[tuple[Path, Path]]]:
    """按固定种子选 train；负样本可检验没有目标时的码率。"""
    if positive_per_task < 1 or negative_per_task < 0:
        raise ValueError("训练样本数无效")
    pairs = list(cityscapes_pairs(root, "train"))
    if not pairs:
        raise ValueError("Cityscapes train 为空")
    random.Random(seed).shuffle(pairs)
    positive: dict[str, list[tuple[Path, Path]]] = {task: [] for task in TARGETS}
    negative: dict[str, list[tuple[Path, Path]]] = {task: [] for task in TARGETS}
    for pair in pairs:
        with Image.open(pair[1]) as source:
            raw = np.asarray(source, dtype=np.uint8)
        for task, (_, raw_id) in TARGETS.items():
            destination = positive[task] if np.any(raw == raw_id) else negative[task]
            limit = positive_per_task if destination is positive[task] else negative_per_task
            if len(destination) < limit:
                destination.append(pair)
        if all(len(positive[task]) >= positive_per_task and
               len(negative[task]) >= negative_per_task for task in TARGETS):
            break
    short = {task: (len(positive[task]), len(negative[task]))
             for task in TARGETS
             if len(positive[task]) != positive_per_task or
             len(negative[task]) != negative_per_task}
    if short:
        raise ValueError(f"Cityscapes train 选样不足（正/负）：{short}")
    return {task: sorted(positive[task] + negative[task]) for task in TARGETS}


def _control_bytes(config: ReplayConfig, task: str, image: Image.Image,
                   selected: np.ndarray, *, first_in_session: bool) -> int:
    """B 仅每任务第一帧收费，M 每张图收费；与 val 字节账本一致。"""
    clip = config.section("models")["clip"]
    bin_name = "person" if task == "pedestrian" else task
    session = SessionHeader(task, (bin_name,), clip["repo"], clip["revision"]).to_bytes()
    patch = int(config.section("image")["patch_size"])
    frame = FrameHeader(image.width, image.height, patch, selected).to_bytes()
    return len(frame) + (len(session) if first_in_session else 0)


def _candidate_rows(search: dict | None, candidates: tuple[Candidate, ...]) -> list[dict]:
    """每个候选保留预算内加权失真最小的质量档；无可行档记失败。"""
    metrics = [] if search is None else search["candidate_metrics"]
    rows = []
    for index, candidate in enumerate(candidates, 1):
        options = [item for item in metrics if item["profile_index"] == index]
        if options:
            best = min(options, key=lambda item: (
                item["weighted_rgb_mse"],
                item["roi_rgb_mse"] if item["roi_rgb_mse"] is not None
                else item["weighted_rgb_mse"],
                item["codec_wire_bytes"],
            ))
            rows.append({"candidate": candidate.name, "feasible": True,
                         "jpeg_quality": best["jpeg_quality"],
                         "wire_bytes": best["codec_wire_bytes"],
                         "roi_rgb_mse": best["roi_rgb_mse"],
                         "background_rgb_mse": best["background_rgb_mse"],
                         "weighted_rgb_mse": best["weighted_rgb_mse"]})
        else:
            rows.append({"candidate": candidate.name, "feasible": False,
                         "jpeg_quality": None, "wire_bytes": None,
                         "roi_rgb_mse": None, "background_rgb_mse": None,
                         "weighted_rgb_mse": None})
    return rows


def summarize_candidates(rows: list[dict], candidates: tuple[Candidate, ...],
                         samples_per_task: int) -> tuple[dict, dict, bool]:
    """可行率先于区域误差；不可行不能靠只统计成功图获得虚高分。"""
    if samples_per_task < 1:
        raise ValueError("每任务至少一张训练图")
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    if len(candidate_by_name) != len(candidates):
        raise ValueError("候选名称重复")
    summary: dict[str, dict] = {}
    selected: dict[str, dict | None] = {}
    for task in TARGETS:
        task_rows = [row for row in rows if row["task"] == task]
        ids = {row["image_id"] for row in task_rows}
        if len(ids) != samples_per_task:
            raise ValueError(f"{task} 训练样本不完整：{len(ids)}/{samples_per_task}")
        task_summary = {}
        for candidate in candidates:
            records = [entry for row in task_rows for entry in row["candidates"]
                       if entry["candidate"] == candidate.name]
            if len(records) != samples_per_task:
                raise ValueError(f"{task}/{candidate.name} 候选结果不完整")
            feasible = [entry for entry in records if entry["feasible"]]
            positive_roi = [entry["roi_rgb_mse"] for entry in feasible
                            if entry["roi_rgb_mse"] is not None]
            task_summary[candidate.name] = {
                "feasible_images": len(feasible),
                "total_images": samples_per_task,
                "feasible_rate": len(feasible) / samples_per_task,
                "positive_roi_images": len(positive_roi),
                "mean_roi_rgb_mse_success": (float(np.mean(positive_roi))
                                              if positive_roi else None),
                "mean_weighted_rgb_mse_success": (
                    float(np.mean([entry["weighted_rgb_mse"] for entry in feasible]))
                    if feasible else None),
                "mean_wire_bytes_success": (
                    float(np.mean([entry["wire_bytes"] for entry in feasible]))
                    if feasible else None),
            }
        summary[task] = task_summary
        # 缺图优先惩罚；同等覆盖时选 ROI 完好且背景改动较小的参数。
        ranking = sorted(candidates, key=lambda candidate: (
            -task_summary[candidate.name]["feasible_images"],
            (task_summary[candidate.name]["mean_roi_rgb_mse_success"]
             if task_summary[candidate.name]["mean_roi_rgb_mse_success"] is not None
             else float("inf")),
            (task_summary[candidate.name]["mean_weighted_rgb_mse_success"]
             if task_summary[candidate.name]["mean_weighted_rgb_mse_success"] is not None
             else float("inf")),
            candidate.subsampling, candidate.background_low_frequency_strength,
        ))
        winner = ranking[0]
        info = task_summary[winner.name]
        selected[task] = ({"candidate": winner.name, "profile": asdict(winner.profile()),
                           "subsampling": winner.subsampling,
                           "train_feasible_rate": info["feasible_rate"],
                           "train_mean_roi_rgb_mse_success":
                           info["mean_roi_rgb_mse_success"]}
                          if info["feasible_images"] else None)
    ready = all(selected[task] is not None and
                selected[task]["train_feasible_rate"] == 1.0 for task in TARGETS)
    return summary, selected, ready


def tune_gt_30kb_sender(config: ReplayConfig, *, output: str | Path,
                        positive_per_task: int = 4, negative_per_task: int = 1,
                        low_frequency_strengths: tuple[float, ...] =
                        (16.0, 64.0, 256.0),
                        subsamplings: tuple[int, ...] = (0,),
                        high_frequency_strength: float = 8.0,
                        zero_threshold_multiplier: float = 2.0,
                        budget_bytes: int = 30_000, seed: int = 2026,
                        max_quality: int = 95) -> dict[str, Any]:
    """冻结每类发送端参数；本函数不加载或调用任何验证集对象。"""
    if (not low_frequency_strengths or
            len(set(low_frequency_strengths)) != len(low_frequency_strengths) or
            any(not np.isfinite(value) or value <= 0 for value in low_frequency_strengths) or
            not subsamplings or len(set(subsamplings)) != len(subsamplings) or
            any(value not in (0, 2) for value in subsamplings) or
            high_frequency_strength < 0 or zero_threshold_multiplier < 0 or
            budget_bytes <= 0 or not 1 <= max_quality <= 100):
        raise ValueError("GT 30 kB 标定参数无效")
    root = config.data_path("cityscapes_root")
    pairs = select_train_pairs(root, positive_per_task=positive_per_task,
                               negative_per_task=negative_per_task, seed=seed)
    output_path = config.path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    cutoff = int(config.section("importance")["dct_high_frequency_cutoff"])
    patch = int(config.section("image")["patch_size"])
    candidates = tuple(Candidate(subsampling, low_strength,
                                 high_frequency_strength,
                                 zero_threshold_multiplier, cutoff)
                       for subsampling in subsamplings
                       for low_strength in low_frequency_strengths)
    settings = {
        "protocol": "figure8a_gt_train_only_30kb_prequant_v1",
        "split": "train", "target_classes": list(TARGETS),
        "positive_per_task": positive_per_task,
        "negative_per_task": negative_per_task,
        "sample_ids_by_task": {task: [path.stem for path, _ in task_pairs]
                               for task, task_pairs in pairs.items()},
        "seed": seed, "budget_wire_bytes": budget_bytes,
        "patch_size": patch, "native_resolution": [2048, 1024],
        "channel_simulated_during_tuning": False,
        "jpeg_restart_markers": False,
        "background_blur": False,
        "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
        "packet_header_bytes": PACKET_HEADER_BYTES,
        "session_header_charge": "first_image_per_task_only",
        "frame_header_charge": "every_image",
        "max_quality": max_quality,
        "roi_weight": 4.0, "background_weight": 1.0,
        "candidate_profiles": [dict(name=candidate.name, **asdict(candidate))
                               for candidate in candidates],
        "config_sha256": sha256(json.dumps(config.values, sort_keys=True,
                                            default=str).encode()).hexdigest(),
    }
    settings_path = output_path / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("已有标定目录的设置不同，必须另选输出目录")
    else:
        write_json(settings_path, settings)
    runs_path = output_path / "runs.jsonl"
    completed: dict[tuple[str, str, int], dict] = {}
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = row["task"], row["image_id"], row["subsampling"]
                if key in completed:
                    raise ValueError(f"标定记录重复：{key}")
                completed[key] = row

    importance_cfg = config.section("importance")
    for task, task_pairs in pairs.items():
        class_id = TARGETS[task][0]
        for image_index, pair in enumerate(task_pairs):
            image_id = pair[0].stem
            if all((task, image_id, subsampling) in completed
                   for subsampling in subsamplings):
                continue
            image, truth = load_cityscapes_pair(*pair)
            if image.size != (2048, 1024):
                raise ValueError(f"Cityscapes 非原始分辨率：{pair[0]}")
            selected = ground_truth_patches(truth, class_id, patch)
            control_bytes = _control_bytes(config, task, image, selected,
                                           first_in_session=image_index == 0)
            for subsampling in subsamplings:
                key = task, image_id, subsampling
                if key in completed:
                    continue
                subset = tuple(candidate for candidate in candidates
                               if candidate.subsampling == subsampling)
                try:
                    _, search = search_task_jpeg(
                        image, selected, budget_bytes=budget_bytes,
                        patch_size=patch, selected_patch_mask=selected,
                        roi_weight=4.0, background_weight=1.0,
                        profiles=(SearchProfile(0.0),
                                  *(candidate.profile() for candidate in subset)),
                        max_quality=max_quality, subsampling=subsampling,
                        cutoff=cutoff,
                        gamma=float(importance_cfg["prequant_gamma"]),
                        importance_ratio_low=float(importance_cfg["importance_ratio_low"]),
                        importance_ratio_high=float(importance_cfg["importance_ratio_high"]),
                        packet_payload_bytes=PACKET_PAYLOAD_BYTES,
                        packet_header_bytes=PACKET_HEADER_BYTES,
                        fixed_control_bytes=control_bytes)
                except ValueError as error:
                    if "无法满足" not in str(error):
                        raise
                    search = None
                row = {
                    "task": task, "image_id": image_id,
                    "target_pixels": int(np.count_nonzero(truth == class_id)),
                    "gt_selected_patch_count": int(selected.sum()),
                    "subsampling": subsampling,
                    "semantic_control_bytes": control_bytes,
                    "candidates": _candidate_rows(search, subset),
                }
                with runs_path.open("a", encoding="utf-8") as destination:
                    destination.write(json.dumps(row, ensure_ascii=False,
                                                 allow_nan=False) + "\n")
                completed[key] = row
            image.close()
            print(f"[train 30 kB sender] {task} {image_index + 1}/{len(task_pairs)}",
                  flush=True)
    rows = list(completed.values())
    summary, selected, ready = summarize_candidates(
        rows, candidates, positive_per_task + negative_per_task)
    report = {"settings": settings, "summary": summary,
              "selected_by_task": selected,
              "all_train_samples_feasible": ready,
              "selection_rule": "maximum full-sample feasibility, then minimum positive ROI MSE; train only",
              "limitation": "训练集小样本标定不保证 val 全部 30 kB 可行；ROI MSE 不是下游 IoU。"}
    write_json(output_path / "frozen_profiles.json", report)
    return report
