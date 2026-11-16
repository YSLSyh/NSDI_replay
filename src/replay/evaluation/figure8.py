"""Cityscapes 图 8(a) 的单任务、GT 重点区域和 30 KB 协议原型。"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from replay.core.config import ReplayConfig, write_json
from replay.data.cityscapes import PAPER_TARGETS, cityscapes_pairs, load_cityscapes_pair


def ground_truth_patches(labels: np.ndarray, class_id: int, patch: int) -> np.ndarray:
    """只要原始 GT 的 patch 内出现目标类别，就标记为重点区域。"""
    height, width = labels.shape
    target = labels == class_id
    padded = np.pad(target, ((0, (-height) % patch), (0, (-width) % patch)), constant_values=False)
    return padded.reshape((height + patch - 1) // patch, patch,
                          (width + patch - 1) // patch, patch).any(axis=(1, 3))


def binary_counts(prediction: np.ndarray, truth: np.ndarray, class_id: int) -> tuple[int, int, int]:
    valid = truth != 255
    predicted = (prediction == class_id) & valid
    actual = (truth == class_id) & valid
    return (int((predicted & actual).sum()), int((predicted & ~actual).sum()),
            int((~predicted & actual).sum()))


def _iou(counts: np.ndarray) -> float:
    tp, fp, fn = (int(value) for value in counts)
    return tp / (tp + fp + fn) if tp + fp + fn else float("nan")


def evaluate_figure8(config: ReplayConfig, device: torch.device) -> dict:
    """复用论文已公开的 GT 区域和 30 KB；未公开的分割器仍为项目选择。"""
    from replay.evaluation.segmentation import CityscapesSegmenter
    from replay.knowledge.encoders import SharedKB
    from replay.knowledge.store import PosteriorKB
    from replay.receiver.infer import load_receiver, restore
    from replay.transmission.header import SessionHeader, FrameHeader
    from replay.transmission.jpeg import decode_jpeg, encode_under_budget
    from replay.core.assets import verify_model_lock
    verify_model_lock(config, "segmenter")
    image_cfg = config.section("image")
    budget = int(config.section("evaluation")["figure8_budget_bytes"])
    patch = int(image_cfg["patch_size"])
    shared = SharedKB.from_config(config, device)
    receiver = load_receiver(config, device)
    segmenter = CityscapesSegmenter(config.model_path("segmenter"), device,
                                     int(image_cfg["segformer_tile"]), int(image_cfg["segformer_stride"]))
    pairs = list(cityscapes_pairs(config.data_path("cityscapes_root"), "val"))
    if not pairs:
        raise ValueError("Cityscapes val 清单为空")
    clip = config.section("models")["clip"]
    report = {"protocol": "figure8_gt_focus_30kb", "split": "val", "num_images": len(pairs),
              "budget_bytes": budget, "tasks": {},
              "limitation": "下游分割器和 KDC 预量化公式未由作者公开；无法与论文图 8 数值直接等同。"}
    for task_name, class_id in PAPER_TARGETS.items():
        kb = PosteriorKB.load(config.data_path("kb_path"))
        bin_name = "person" if class_id == 11 else task_name
        session = SessionHeader(task_name, (bin_name,), clip["repo"], clip["revision"])
        session_bytes = session.to_bytes()
        totals = {name: np.zeros(3, np.int64) for name in ("original", "task_jpeg", "restored")}
        sizes = []
        for frame_number, (image_path, label_path) in enumerate(tqdm(pairs, desc=f"Figure 8 {task_name}")):
            image, truth = load_cityscapes_pair(image_path, label_path)
            selected = ground_truth_patches(truth, class_id, patch)
            frame = FrameHeader(image.width, image.height, patch, selected)
            frame_bytes = frame.to_bytes()
            # B 在会话开始只计费一次；每张图都计入逐帧 M。
            charged_session_bytes = len(session_bytes) if frame_number == 0 else 0
            jpeg_budget = budget - charged_session_bytes - len(frame_bytes)
            if jpeg_budget <= 0:
                raise ValueError("语义 header 已超过 30 KB 总预算")
            importance_cfg = config.section("importance")
            payload, encode_info = encode_under_budget(
                image, selected, patch, jpeg_budget, int(image_cfg["jpeg_subsampling"]),
                int(importance_cfg["dct_high_frequency_cutoff"]),
                luma_strength=float(importance_cfg["prequant_luma_strength"]),
                chroma_strength=float(importance_cfg["prequant_chroma_strength"]),
                gamma=float(importance_cfg["prequant_gamma"]),
                importance_ratio_low=float(importance_cfg["importance_ratio_low"]),
                importance_ratio_high=float(importance_cfg["importance_ratio_high"]))
            transmitted = decode_jpeg(payload)
            restored = restore(transmitted, list(session.bins), shared, kb, receiver,
                               int(config.section("knowledge")["rx_top_k"]), device,
                               merge_cosine=float(config.section("knowledge")["session_merge_cosine"]))
            for name, candidate in (("original", image), ("task_jpeg", transmitted), ("restored", restored)):
                totals[name] += np.asarray(binary_counts(segmenter.predict(candidate), truth, class_id))
            sizes.append({"image": image_path.name, "jpeg_bytes": len(payload),
                          "session_header_bytes_charged": charged_session_bytes,
                          "frame_header_bytes": len(frame_bytes),
                          "total_bytes": len(payload) + charged_session_bytes + len(frame_bytes), **encode_info})
        report["tasks"][task_name] = {"iou": {name: _iou(value) for name, value in totals.items()},
                                      "bytes": sizes, "selected_region_source": "Cityscapes ground truth"}
        write_json(config.root / "artifacts" / "figure8_gt_focus_30kb.json", report)
    return report
