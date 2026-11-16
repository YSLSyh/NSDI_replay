"""在单个标准 JPEG 码流中搜索任务感知预量化参数。

论文给出按重要性调节预量化与最终标准 JPEG 编码的方向，但没有给出参数搜索。
这里仅用发送端已有的图像和重要性图评估候选；不读取验证集类别标签。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np
from PIL import Image

from replay.transmission.jpeg import decode_jpeg, encode_task_jpeg
from replay.transmission.plain_jpeg_channel import plain_jpeg_wire_bytes


@dataclass(frozen=True)
class SearchProfile:
    """一组预量化参数；相同亮度、色度强度避免额外色偏。"""

    strength: float
    zero_threshold_multiplier: float = 0.0
    protected_selected: bool = False
    cutoff: int | None = None
    background_low_frequency_strength: float = 0.0
    # None 保持旧行为：色度低频使用与亮度相同的强度。
    background_chroma_low_frequency_strength: float | None = None


DEFAULT_PROFILES = (
    SearchProfile(0.0, 0.0),  # 原始标准 JPEG 兜底候选
    SearchProfile(2.0, 0.0),
    SearchProfile(2.0, 0.0, True, 4),
    SearchProfile(4.0, 0.0, True, 4),
    SearchProfile(6.0, 0.0, True, 4),
    SearchProfile(2.0, 0.0, True, 2),
    SearchProfile(4.0, 0.0, True, 2),
    SearchProfile(4.0, 0.35, True, 2),
)


def _wire_bytes(payload: bytes, packet_payload_bytes: int,
                packet_header_bytes: int, fixed_control_bytes: int) -> int:
    return (plain_jpeg_wire_bytes(payload, packet_payload_bytes=packet_payload_bytes,
                                  packet_header_bytes=packet_header_bytes) +
            fixed_control_bytes)


def _selected_mask(importance: np.ndarray, image: Image.Image, patch_size: int,
                   selected_patch_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int, str]:
    expected = (math.ceil(image.height / patch_size),
                math.ceil(image.width / patch_size))
    alpha = np.asarray(importance)
    if (alpha.shape != expected or
            not (alpha.dtype == np.bool_ or np.issubdtype(alpha.dtype, np.number))):
        raise ValueError(f"重要性图应为 {expected} 数值或布尔 patch 网格")
    alpha = alpha.astype(np.float64)
    if not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("重要性图必须是有限非负数")
    if selected_patch_mask is None:
        # 论文的 1/N 均匀权重阈值；未选到任何 patch 时退化为全图均匀评价。
        total = float(alpha.sum())
        patches = alpha > total / alpha.size if total else np.zeros(expected, dtype=bool)
        source = "alpha_above_mean"
    else:
        patches = np.asarray(selected_patch_mask)
        if patches.shape != expected or patches.dtype != np.bool_:
            raise ValueError(f"selected_patch_mask 应为 {expected} 布尔 patch 网格")
        source = "selected_patch_mask"
    count = int(patches.sum())
    pixels = np.repeat(np.repeat(patches, patch_size, axis=0),
                       patch_size, axis=1)[:image.height, :image.width]
    return pixels, patches, count, source


def _distortion(original: np.ndarray, reconstructed: Image.Image,
                selected_pixels: np.ndarray, roi_weight: float,
                background_weight: float) -> tuple[float, float | None, float | None]:
    restored = np.asarray(reconstructed, dtype=np.float32)
    # 对 RGB 三通道求均方误差；同一张输入图上所有候选共享同一权重掩码。
    pixel_error = np.mean((original - restored) ** 2, axis=2)
    roi_error = float(pixel_error[selected_pixels].mean()) if selected_pixels.any() else None
    background = ~selected_pixels
    bg_error = float(pixel_error[background].mean()) if background.any() else None
    # 区域内先各自平均，再按类别权重组合。小目标不会被背景像素数量淹没。
    if roi_error is None:
        score = bg_error
    elif bg_error is None:
        score = roi_error
    else:
        score = ((roi_weight * roi_error + background_weight * bg_error) /
                 (roi_weight + background_weight))
    assert score is not None
    return score, roi_error, bg_error


def search_task_jpeg(
    image: Image.Image,
    importance: np.ndarray,
    *,
    budget_bytes: int,
    patch_size: int = 16,
    selected_patch_mask: np.ndarray | None = None,
    roi_weight: float = 4.0,
    background_weight: float = 1.0,
    profiles: Sequence[SearchProfile] = DEFAULT_PROFILES,
    max_quality: int = 95,
    subsampling: int = 0,
    cutoff: int = 4,
    gamma: float = 1.0,
    importance_ratio_low: float = 1.0,
    importance_ratio_high: float = 4.0,
    packet_payload_bytes: int = 1200,
    packet_header_bytes: int = 8,
    fixed_control_bytes: int = 0,
    optimize_huffman: bool = True,
    require_plain_fallback: bool = True,
) -> tuple[bytes, dict]:
    """返回预算内加权失真最低的单个 JPEG 码流及可审计的选择记录。

    质量搜索在每组预量化参数的预算边界附近进行。JPEG 结构标记可靠传输，
    仅扫描熵数据按固定大小分片计包头；本函数不模拟网络丢包或恢复。
    """
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("image 必须是非空 PIL 图像")
    if (type(budget_bytes) is not int or budget_bytes <= 0 or
            type(patch_size) is not int or patch_size <= 0 or patch_size % 8 or
            type(max_quality) is not int or not 1 <= max_quality <= 100 or
            type(packet_payload_bytes) is not int or packet_payload_bytes <= 0 or
            type(packet_header_bytes) is not int or packet_header_bytes < 0 or
            type(fixed_control_bytes) is not int or fixed_control_bytes < 0 or
            type(require_plain_fallback) is not bool or
            subsampling not in (-1, 0, 1, 2) or
            not all(math.isfinite(value) and value > 0 for value in
                    (roi_weight, background_weight)) or
            not profiles):
        raise ValueError("搜索参数或预算无效")
    if any(not isinstance(profile, SearchProfile) or
           not math.isfinite(profile.strength) or profile.strength < 0 or
           not math.isfinite(profile.zero_threshold_multiplier) or
           profile.zero_threshold_multiplier < 0 or
           not math.isfinite(profile.background_low_frequency_strength) or
           profile.background_low_frequency_strength < 0 or
           (profile.background_chroma_low_frequency_strength is not None and
            (not math.isfinite(profile.background_chroma_low_frequency_strength) or
             profile.background_chroma_low_frequency_strength < 0)) or
           type(profile.protected_selected) is not bool or
           (profile.cutoff is not None and
            (type(profile.cutoff) is not int or not 0 <= profile.cutoff <= 14))
           for profile in profiles):
        raise ValueError("预量化候选参数无效")
    if require_plain_fallback and not any(
            profile.strength == 0 and profile.zero_threshold_multiplier == 0 and
            profile.background_low_frequency_strength == 0 and
            profile.background_chroma_low_frequency_strength in (None, 0) and
            not profile.protected_selected for profile in profiles):
        raise ValueError("候选参数必须包含普通 JPEG 兜底候选")

    rgb = image.convert("RGB")
    selected, selected_patches, selected_count, mask_source = _selected_mask(
        importance, rgb, patch_size, selected_patch_mask)
    original = np.asarray(rgb, dtype=np.float32)
    cache: dict[tuple[int, int], bytes] = {}
    evaluated: list[dict] = []
    best: tuple[tuple[float, float, int], bytes, dict] | None = None

    def encoded(profile_index: int, quality: int) -> bytes:
        key = (profile_index, quality)
        if key not in cache:
            profile = profiles[profile_index]
            cache[key] = encode_task_jpeg(
                rgb, importance, patch_size, quality, subsampling,
                profile.cutoff if profile.cutoff is not None else cutoff,
                luma_strength=profile.strength,
                chroma_strength=profile.strength,
                gamma=gamma,
                importance_ratio_low=importance_ratio_low,
                importance_ratio_high=importance_ratio_high,
                background_lowpass_sigma=0.0,
                fast_prequant=True,
                zero_threshold_multiplier=profile.zero_threshold_multiplier,
                background_low_frequency_strength=profile.background_low_frequency_strength,
                background_chroma_low_frequency_strength=profile.background_chroma_low_frequency_strength,
                roi_halo_pixels=0.0,
                restart_marker_rows=0,
                restart_marker_blocks=0,
                optimize_huffman=optimize_huffman,
                protected_patch_mask=(selected_patches if profile.protected_selected
                                      else None))
        return cache[key]

    def fits(profile_index: int, quality: int) -> bool:
        return (_wire_bytes(encoded(profile_index, quality), packet_payload_bytes,
                            packet_header_bytes, fixed_control_bytes) <= budget_bytes)

    for profile_index, profile in enumerate(profiles):
        if not fits(profile_index, 1):
            continue
        # libjpeg 的 optimize 后码流长度大体随 Q 增长；边界附近再逐档复核。
        low, high, frontier = 2, max_quality, 1
        while low <= high:
            middle = (low + high) // 2
            if fits(profile_index, middle):
                frontier = middle
                low = middle + 1
            else:
                high = middle - 1
        qualities = {frontier}
        qualities.update(range(max(1, frontier - 3),
                               min(max_quality, frontier + 3) + 1))
        for quality in sorted(qualities):
            payload = encoded(profile_index, quality)
            wire = _wire_bytes(payload, packet_payload_bytes,
                               packet_header_bytes, fixed_control_bytes)
            if wire > budget_bytes:
                continue
            restored = decode_jpeg(payload)
            score, roi_mse, background_mse = _distortion(
                original, restored, selected, roi_weight, background_weight)
            record = {
                "profile_index": profile_index,
                "jpeg_quality": quality,
                "luma_strength": profile.strength,
                "chroma_strength": profile.strength,
                "zero_threshold_multiplier": profile.zero_threshold_multiplier,
                "background_low_frequency_strength": profile.background_low_frequency_strength,
                "background_chroma_low_frequency_strength": (
                    profile.background_low_frequency_strength
                    if profile.background_chroma_low_frequency_strength is None
                    else profile.background_chroma_low_frequency_strength),
                "protected_selected": profile.protected_selected,
                "cutoff": profile.cutoff if profile.cutoff is not None else cutoff,
                "jpeg_bytes": len(payload),
                "codec_wire_bytes": wire,
                "weighted_rgb_mse": score,
                "roi_rgb_mse": roi_mse,
                "background_rgb_mse": background_mse,
                "plain_jpeg": (profile.strength == 0 and
                               profile.zero_threshold_multiplier == 0 and
                               profile.background_low_frequency_strength == 0 and
                               profile.background_chroma_low_frequency_strength in (None, 0)),
            }
            evaluated.append(record)
            # 相同失真时优先更低 ROI 误差，再优先更小实际传输字节数。
            rank = (score, roi_mse if roi_mse is not None else score, wire)
            if best is None or rank < best[0]:
                best = (rank, payload, record)
    if best is None:
        raise ValueError(f"原尺寸单 JPEG 无法满足 {budget_bytes} 字节总传输预算")

    result = dict(best[2])
    result.update({
        "selection_method": "sender_weighted_rgb_mse_search_project_choice",
        "selection_uses_validation_labels": False,
        "semantic_mask_source": mask_source,
        "selected_patch_count": selected_count,
        "total_patch_count": int(np.asarray(importance).size),
        "roi_weight": roi_weight,
        "background_weight": background_weight,
        "candidate_profiles": len(profiles),
        "require_plain_fallback": require_plain_fallback,
        "evaluated_feasible_candidates": len(evaluated),
        "budget_bytes": budget_bytes,
        "packet_payload_bytes": packet_payload_bytes,
        "packet_header_bytes": packet_header_bytes,
        "fixed_control_bytes": fixed_control_bytes,
        "restart_marker_rows": 0,
        "restart_marker_blocks": 0,
        "subsampling": subsampling,
        "cutoff": cutoff,
        "gamma": gamma,
        "importance_ratio_low": importance_ratio_low,
        "importance_ratio_high": importance_ratio_high,
        "optimize_huffman": optimize_huffman,
        "fast_prequant": True,
        "candidate_metrics": evaluated,
    })
    return best[1], result

