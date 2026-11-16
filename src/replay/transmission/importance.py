"""CLIPSeg 逐块提议区域，再用 CLIP 计算语义权重。"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from replay.core.config import ReplayConfig
if TYPE_CHECKING:
    from replay.knowledge.encoders import SharedKB


def select_patches(scores: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("token_keep_fraction 必须在 (0, 1] 内")
    flat = scores.reshape(-1)
    count = max(1, int(np.ceil(flat.size * fraction)))
    selected = np.zeros(flat.size, dtype=bool)
    selected[np.argsort(-flat, kind="stable")[:count]] = True
    return selected.reshape(scores.shape)


def paper_token_importance(tokens: np.ndarray, bin_vectors: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    """论文通用 token 公式 Eq. (3)；图像区域重要性另走 RegionImportance。"""
    maximum = paper_max_cosine(tokens, bin_vectors)
    logits = maximum.reshape(-1) / temperature
    weights = np.exp(logits - logits.max())
    return (weights / weights.sum()).reshape(maximum.shape)


def paper_max_cosine(tokens: np.ndarray, bin_vectors: np.ndarray) -> np.ndarray:
    """Eq. (3) 的 token/文本余弦；不能单独作为图像区域重要性图。"""
    return np.asarray((tokens @ bin_vectors.T).max(axis=-1), dtype=np.float32)


def _clipseg_inputs(processor, tile: Image.Image, queries: list[str]):
    """显式指定 RGB 图块的通道位置，避免 1 像素边缘块被误判为单通道。"""
    # CLIPSegProcessor 会把额外参数同时传给 tokenizer，故分别处理文本和图像。
    inputs = processor(text=queries, padding=True, return_tensors="pt")
    pixels = processor.image_processor(
        images=[tile] * len(queries), return_tensors="pt",
        input_data_format="channels_last",
    )
    inputs["pixel_values"] = pixels["pixel_values"]
    return inputs


class RegionImportance:
    def __init__(self, config: ReplayConfig, device: torch.device, shared: SharedKB):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        from replay.core.assets import verify_model_lock
        verify_model_lock(config, "clipseg")
        self.config, self.device, self.shared = config, device, shared
        path = config.model_path("clipseg")
        self.processor = CLIPSegProcessor.from_pretrained(path, local_files_only=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained(path, local_files_only=True).to(device).eval()

    @torch.inference_mode()
    def region_proposals(self, image: Image.Image, bins: list[str]) -> np.ndarray:
        """CLIPSeg 提区域，再比较区域特征与各语义桶，生成区域重要性图。"""
        if not bins:
            raise ValueError("语义类别不能为空")
        image = image.convert("RGB")
        width, height = image.size
        size = int(self.config.section("image")["clipseg_tile"])
        stride = int(self.config.section("image")["clipseg_stride"])
        if stride != size:
            raise ValueError("当前实现要求无重叠 CLIPSeg 块；这样不会对原图做全局缩放")
        heat = np.zeros((height, width), np.float32)
        importance = self.config.section("importance")
        probability = float(importance["region_probability"])
        alias_relative = importance.get("visual_alias_relative_threshold")
        alias_floor = float(importance.get("visual_alias_min_probability", probability))
        alias_min_peak = float(importance.get("visual_alias_min_peak", 0.0))
        temperature = float(importance["temperature"])
        min_pixels = int(importance.get("region_min_pixels", 16))
        aliases = importance.get("clipseg_query_aliases", {})
        groups = importance.get("visual_query_groups", {})
        if not isinstance(groups, dict):
            raise ValueError("visual_query_groups 必须是类别到视觉查询词列表的映射")
        queries: list[str] = []
        grouped_queries: dict[str, list[str]] = {}
        comparison_terms = list(bins)
        for name in bins:
            group = groups.get(name)
            if group is None:
                # 旧配置保持原行为：替换 CLIPSeg 查询词，但区域仍与标准桶名比较。
                queries.append(aliases.get(name, name))
                continue
            if not isinstance(group, list) or not group or not all(
                    isinstance(term, str) and term.strip() for term in group):
                raise ValueError(f"visual_query_groups.{name} 必须是非空词语列表")
            # 一个语义桶可以包含多种视觉名称；候选和文本相似度均覆盖这些名称。
            queries.extend(group)
            grouped_queries[name] = list(dict.fromkeys(group))
            comparison_terms.extend(group)
        queries = list(dict.fromkeys(queries))
        group_aliases = {term for name, terms in grouped_queries.items()
                         for term in terms if term != name}
        comparison_terms = list(dict.fromkeys(comparison_terms))
        text_vectors = self.shared.text(comparison_terms)
        candidates: list[tuple[int, int, np.ndarray, float, str]] = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                tile = image.crop((x, y, min(x + size, width), min(y + size, height)))
                inputs = _clipseg_inputs(self.processor, tile, queries)
                logits = self.model(**{key: value.to(self.device) for key, value in inputs.items()}).logits
                logits = F.interpolate(logits[:, None], size=(tile.height, tile.width),
                                       mode="bilinear", align_corners=False)[:, 0].sigmoid().cpu().numpy()
                # 每个查询的二值掩码拆成连通区域，避免远处目标与路面共用一个大裁剪框。
                for query, mask in zip(queries, logits):
                    query_probability = probability
                    if query in group_aliases and alias_relative is not None and float(mask.max()) >= alias_min_peak:
                        # 视觉别名的置信度可能低于桶原名；仅对有强种子的别名放宽阈值。
                        query_probability = max(alias_floor, min(probability, float(mask.max()) * float(alias_relative)))
                    binary = mask >= query_probability
                    if not binary.any():
                        continue
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(
                        binary.astype(np.uint8), connectivity=8)
                    for component_id in range(1, count):
                        left, top, crop_width, crop_height, area = (
                            int(value) for value in stats[component_id])
                        if area < min_pixels:
                            continue
                        right, bottom = left + crop_width, top + crop_height
                        component = labels[top:bottom, left:right] == component_id
                        # 区域外填中性灰，避免大矩形内的道路/建筑主导 CLIP 区域特征。
                        crop_pixels = np.asarray(tile.crop((left, top, right, bottom))).copy()
                        crop_pixels[~component] = 127
                        crop = Image.fromarray(crop_pixels, "RGB")
                        if crop.width < 16 or crop.height < 16:
                            padded = Image.new("RGB", (max(16, crop.width), max(16, crop.height)),
                                               (127, 127, 127))
                            padded.paste(crop)
                            crop = padded
                        # 图像路径先用 CLIPSeg 得到连通区域，再用 CLIP 区域特征与文本比较；
                        # 不直接拿 ViT 的原始 patch token 当作区域特征。
                        # 按论文 Eq.(3) 取最大余弦；组合桶还比较其中的视觉名称。
                        region_vector = self.shared.images([crop])[0]
                        similarity = float((region_vector @ text_vectors.T).max())
                        confidence = np.where(component, mask[top:bottom, left:right], 0.0).astype(np.float32)
                        candidates.append((y + top, x + left, confidence, similarity, query))
        if not candidates:
            return heat
        # 论文给定 T_attn=0.07；区域候选的融合范围没有公开。
        similarities = np.asarray([candidate[3] for candidate in candidates], dtype=np.float32)
        logits = (similarities - similarities.max()) / temperature
        weights = np.exp(logits)
        weights /= weights.sum()
        if importance.get("balance_visual_queries", False):
            # 只在组合桶内部平衡视觉名称，并保持它在全局 softmax 中的总权重。
            for group in grouped_queries.values():
                if len(group) < 2:
                    continue
                group_indices = [i for i, candidate in enumerate(candidates) if candidate[4] in group]
                if not group_indices:
                    continue
                group_mass = float(weights[group_indices].sum())
                balanced = np.zeros(len(group_indices), dtype=np.float32)
                for query in group:
                    positions = [j for j, i in enumerate(group_indices) if candidates[i][4] == query]
                    if not positions:
                        continue
                    scores = similarities[[group_indices[j] for j in positions]]
                    balanced[positions] = np.exp((scores - scores.max()) / temperature)
                weights[group_indices] = group_mass * balanced / balanced.sum()
        for (y, x, confidence, _, _), weight in zip(candidates, weights):
            window = heat[y:y + confidence.shape[0], x:x + confidence.shape[1]]
            # 重叠区域取最大贡献；论文未公开像素融合规则，这是项目实现。
            np.maximum(window, confidence * weight, out=window)
        return heat

    @torch.inference_mode()
    def map(self, image: Image.Image, bins: list[str],
            return_scores: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = image.convert("RGB")
        width, height = image.size
        heat = self.region_proposals(image, bins)
        patch = int(self.config.section("image")["patch_size"])
        pad_h = (-height) % patch
        pad_w = (-width) % patch
        padded = np.pad(heat, ((0, pad_h), (0, pad_w)), mode="constant")
        region_scores = padded.reshape((height + pad_h) // patch, patch, (width + pad_w) // patch, patch).mean((1, 3))
        # 区域图聚合到 16×16 patch 后归一化，再按论文 Eq. (2) 的 alpha_i >= tau 选 token。
        # 论文没有公开 tau 的默认值；这里以均匀分配 1/N 为可审计的项目阈值。
        total = float(region_scores.sum())
        scores = region_scores / total if total > 0 else np.zeros_like(region_scores)
        tau = 1.0 / scores.size
        selected = (scores >= tau) & (region_scores > 0)
        if return_scores:
            return heat, selected, scores
        return heat, selected
