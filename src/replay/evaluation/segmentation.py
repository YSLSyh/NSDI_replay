"""以原始图像尺寸做滑窗分割，计算官方 19 类 IoU。"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

from replay.core.config import ReplayConfig, write_json
from replay.data.cityscapes import cityscapes_pairs, load_cityscapes_pair, TRAIN_NAMES, PAPER_TARGETS
from replay.knowledge.bins import read_bins
from replay.knowledge.encoders import SharedKB
from replay.knowledge.store import PosteriorKB
from replay.receiver.infer import load_receiver, restore
from replay.transmission.importance import RegionImportance
from replay.transmission.jpeg import encode_task_jpeg, decode_jpeg


def confusion(prediction: np.ndarray, target: np.ndarray, classes: int = 19) -> np.ndarray:
    valid = (target != 255) & (target >= 0) & (target < classes)
    return np.bincount(classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64),
                       minlength=classes * classes).reshape(classes, classes)


def scores(matrix: np.ndarray) -> dict:
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    per_class = np.divide(intersection, union, out=np.full(19, np.nan), where=union > 0)
    return {"mIoU": float(np.nanmean(per_class)),
            "per_class_IoU": {name: float(per_class[i]) for i, name in enumerate(TRAIN_NAMES)},
            "paper_targets_IoU": {name: float(per_class[index]) for name, index in PAPER_TARGETS.items()}}


class CityscapesSegmenter:
    def __init__(self, model_path: Path, device: torch.device, tile: int, stride: int):
        self.device, self.tile, self.stride = device, tile, stride
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path, local_files_only=True).to(device).eval()
        if not 0 < stride <= tile:
            raise ValueError("语义分割滑窗步长必须在 (0, tile] 内")

    @staticmethod
    def _starts(length: int, tile: int, stride: int) -> list[int]:
        """最后一块贴齐图像边缘，避免把 Cityscapes 原图缩放或留下窄边块。"""
        end = max(0, length - tile)
        starts = list(range(0, end + 1, stride))
        if starts[-1] != end:
            starts.append(end)
        return starts

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        width, height = image.size
        logits_sum = np.zeros((int(self.model.config.num_labels), height, width), dtype=np.float32)
        visits = np.zeros((height, width), dtype=np.float32)
        for y in self._starts(height, self.tile, self.stride):
            for x in self._starts(width, self.tile, self.stride):
                patch = image.crop((x, y, min(x + self.tile, width), min(y + self.tile, height)))
                # 禁用处理器默认 resize；只把原始分辨率切成推理块。
                inputs = self.processor(images=patch, return_tensors="pt", do_resize=False)
                logits = self.model(**{key: value.to(self.device) for key, value in inputs.items()}).logits
                logits = F.interpolate(logits, size=(patch.height, patch.width), mode="bilinear", align_corners=False)
                logits_sum[:, y:y + patch.height, x:x + patch.width] += logits[0].float().cpu().numpy()
                visits[y:y + patch.height, x:x + patch.width] += 1
        return (logits_sum / visits[None]).argmax(axis=0).astype(np.uint8)


def evaluate_cityscapes(config: ReplayConfig, device: torch.device, split: str = "val") -> dict:
    from replay.core.assets import verify_model_lock
    verify_model_lock(config, "segmenter")
    shared = SharedKB.from_config(config, device)
    kb = PosteriorKB.load(config.data_path("kb_path"))
    bins = read_bins(config.path(config.section("task")["bin_cache"]), config.section("task")["description"])
    importance = RegionImportance(config, device, shared)
    receiver = load_receiver(config, device)
    image_cfg = config.section("image")
    segmenter = CityscapesSegmenter(config.model_path("segmenter"), device,
                                     int(image_cfg["segformer_tile"]), int(image_cfg["segformer_stride"]))
    totals = {name: np.zeros((19, 19), dtype=np.int64) for name in ("original", "jpeg", "task_jpeg", "restored")}
    byte_counts = []
    pairs = list(cityscapes_pairs(config.data_path("cityscapes_root"), split))
    if not pairs:
        raise ValueError(f"Cityscapes {split} 没有样本")
    for image_path, label_path in tqdm(pairs, desc=f"Cityscapes {split}"):
        image, truth = load_cityscapes_pair(image_path, label_path)
        _, selected, alpha = importance.map(image, bins, return_scores=True)
        importance_cfg = config.section("importance")
        stream = encode_task_jpeg(
            image, alpha, int(image_cfg["patch_size"]),
            int(image_cfg["jpeg_quality"]), int(image_cfg["jpeg_subsampling"]),
            int(importance_cfg["dct_high_frequency_cutoff"]),
            luma_strength=float(importance_cfg["prequant_luma_strength"]),
            chroma_strength=float(importance_cfg["prequant_chroma_strength"]),
            gamma=float(importance_cfg["prequant_gamma"]),
            importance_ratio_low=float(importance_cfg["importance_ratio_low"]),
            importance_ratio_high=float(importance_cfg["importance_ratio_high"]))
        raw_jpeg = BytesIO()
        image.save(raw_jpeg, format="JPEG", quality=int(image_cfg["jpeg_quality"]),
                   subsampling=int(image_cfg["jpeg_subsampling"]))
        task_image = decode_jpeg(stream)
        repaired = restore(task_image, bins, shared, kb, receiver,
                           int(config.section("knowledge")["rx_top_k"]), device,
                           merge_cosine=float(config.section("knowledge")["session_merge_cosine"]))
        candidates = {"original": image, "jpeg": decode_jpeg(raw_jpeg.getvalue()),
                      "task_jpeg": task_image, "restored": repaired}
        for name, candidate in candidates.items():
            totals[name] += confusion(segmenter.predict(candidate), truth)
        byte_counts.append({"image": image_path.name, "jpeg": len(raw_jpeg.getvalue()),
                            "task_jpeg": len(stream)})
    report = {"split": split, "num_images": len(pairs),
              "methods": {name: scores(total) for name, total in totals.items()},
              "bytes": byte_counts, "note": "发送端 JPEG 质量固定；本报告未强制 30KB 等码率预算。"}
    write_json(config.root / "artifacts" / f"cityscapes_{split}_results.json", report)
    return report
