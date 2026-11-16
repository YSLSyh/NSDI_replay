"""用连续区域重要性调节 DCT 预量化，再交给标准 JPEG 编码器。"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageFile


_JPEG_SAVE_LOCK = Lock()


def _jpeg_bytes(image: Image.Image, quality: int, subsampling: int,
                restart_marker_rows: int = 0, restart_marker_blocks: int = 0,
                optimize_huffman: bool = False) -> bytes:
    stream = BytesIO()
    if not isinstance(optimize_huffman, bool):
        raise TypeError("optimize_huffman 必须是布尔值")
    options = {"format": "JPEG", "quality": quality, "subsampling": subsampling,
               "optimize": optimize_huffman}
    if restart_marker_rows and restart_marker_blocks:
        raise ValueError("restart_marker_rows 与 restart_marker_blocks 不能同时设置")
    if restart_marker_rows:
        options["restart_marker_rows"] = restart_marker_rows
    if restart_marker_blocks:
        options["restart_marker_blocks"] = restart_marker_blocks
    # Pillow 对 optimize=True 的 JPEG 要求一次容纳整幅码流；默认 64 KiB
    # 缓冲区在高熵图像上会报 "Suspension not allowed here"。仅在本次保存时
    # 扩大缓冲区，并恢复进程原值；不改变质量、采样或码流内容。
    with _JPEG_SAVE_LOCK:
        original_maxblock = ImageFile.MAXBLOCK
        try:
            if optimize_huffman:
                ImageFile.MAXBLOCK = max(original_maxblock, 4 * image.width * image.height)
            image.save(stream, **options)
        finally:
            ImageFile.MAXBLOCK = original_maxblock
    return stream.getvalue()


@lru_cache(maxsize=128)
def _codec_quant_tables(quality: int, subsampling: int) -> tuple[np.ndarray, np.ndarray]:
    """读取当前 Pillow/libjpeg 真正使用的量化表，不另造一套近似表。"""
    sample = Image.new("RGB", (16, 16), (128, 128, 128))
    with Image.open(BytesIO(_jpeg_bytes(sample, quality, subsampling))) as encoded:
        tables = encoded.quantization
    if not tables or 0 not in tables:
        raise ValueError("JPEG 编码器未提供亮度量化表")
    # Pillow 返回自然的 8×8 行优先顺序，两个色度分量通常共用表 1。
    luma = np.asarray(tables[0], dtype=np.float32).reshape(8, 8)
    chroma = np.asarray(tables.get(1, tables[0]), dtype=np.float32).reshape(8, 8)
    return luma, chroma


def _validate_parameters(patch_size: int, quality: int, subsampling: int, cutoff: int,
                         luma_strength: float, chroma_strength: float, gamma: float,
                         importance_ratio_low: float, importance_ratio_high: float,
                         background_low_frequency_strength: float = 0.0,
                         background_chroma_low_frequency_strength: float | None = None) -> None:
    if patch_size <= 0 or patch_size % 8:
        raise ValueError("patch_size 必须是正的 8 的倍数")
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality 必须在 1–100 之间")
    if subsampling not in (-1, 0, 1, 2):
        raise ValueError("JPEG subsampling 必须为 -1、0、1 或 2")
    if not 0 <= cutoff <= 14:
        raise ValueError("DCT 截止频率必须在 0–14 之间")
    values = (luma_strength, chroma_strength, gamma,
              importance_ratio_low, importance_ratio_high,
              background_low_frequency_strength)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("预量化参数必须是有限数值")
    if (background_chroma_low_frequency_strength is not None and
            (not np.isfinite(background_chroma_low_frequency_strength) or
             background_chroma_low_frequency_strength < 0)):
        raise ValueError("色度低频强度必须是有限非负数")
    if (luma_strength < 0 or chroma_strength < 0 or
            background_low_frequency_strength < 0 or gamma <= 0):
        raise ValueError("预量化强度必须非负，gamma 必须为正")
    if importance_ratio_low < 0 or importance_ratio_high <= importance_ratio_low:
        raise ValueError("重要性比例的上界必须大于非负下界")


def _patch_weights(importance: np.ndarray, image: Image.Image, patch_size: int,
                   ratio_low: float, ratio_high: float) -> np.ndarray:
    """把 Eq.(3) 的归一化 alpha 映射为固定标尺权重；布尔输入供 GT oracle 使用。"""
    values = np.asarray(importance)
    expected = ((image.height + patch_size - 1) // patch_size,
                (image.width + patch_size - 1) // patch_size)
    if values.shape != expected:
        raise ValueError(f"patch 重要性形状应为 {expected}，实际 {values.shape}")
    if values.dtype == np.bool_:
        return values.astype(np.float32)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("patch 重要性必须是数值数组或布尔掩码")
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("patch 重要性必须是有限非负数")
    total = values.sum(dtype=np.float64)
    if total == 0:
        return np.zeros(expected, dtype=np.float32)
    # alpha 总和为 1；N*alpha=1 是均匀分配基线。边界在调参集确定，推理时不逐图拉伸。
    relative = values * (values.size / total)
    return np.clip((relative - ratio_low) / (ratio_high - ratio_low),
                   0.0, 1.0).astype(np.float32)


def _background_lowpass(image: Image.Image, weights: np.ndarray,
                        patch_size: int, sigma: float,
                        roi_halo_pixels: float = 0.0) -> Image.Image:
    """只抑制低重要性区域的高频；保持前景与输出尺寸不变。"""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("背景低通 sigma 必须是有限非负数")
    if not np.isfinite(roi_halo_pixels) or roi_halo_pixels < 0:
        raise ValueError("ROI 渐变宽度必须是有限非负数")
    if sigma == 0 or np.all(weights == 1):
        return image
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    blurred = cv2.GaussianBlur(original, (0, 0), sigmaX=sigma, sigmaY=sigma)
    pixel_weights = np.repeat(np.repeat(weights, patch_size, axis=0), patch_size,
                              axis=1)[:image.height, :image.width]
    if roi_halo_pixels and np.all((pixel_weights == 0) | (pixel_weights == 1)):
        # 目标区域原像素保持完整；背景靠近目标处做线性过渡，避免 patch 网格硬接缝。
        outside = (pixel_weights == 0).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
        pixel_weights = np.maximum(pixel_weights,
                                   np.clip(1.0 - distance / roi_halo_pixels,
                                           0.0, 1.0))
    if np.all((pixel_weights == 0) | (pixel_weights == 1)):
        # 布尔关注图精确保留重要区域的每个像素，避免浮点混合改变前景。
        result = np.where(pixel_weights[:, :, None] == 1, original, blurred)
    else:
        blend = pixel_weights[:, :, None]
        result = np.rint(original.astype(np.float32) * blend +
                         blurred.astype(np.float32) * (1 - blend))
        result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result, "RGB")


@lru_cache(maxsize=1)
def _dct_basis() -> np.ndarray:
    """正交 8×8 DCT-II 基矩阵，供显式选择的批量预量化路径使用。"""
    row = np.arange(8)[:, None]
    col = np.arange(8)[None, :]
    basis = np.cos(np.pi * (2 * col + 1) * row / 16).astype(np.float32)
    basis[0] *= np.float32(1 / np.sqrt(8))
    basis[1:] *= np.float32(np.sqrt(2 / 8))
    return basis


def _prequantize_blocks_fast(
        padded: np.ndarray, weights: np.ndarray, patch_size: int,
        frequency_weight: np.ndarray, low_frequency_weight: np.ndarray,
        tables: tuple[np.ndarray, ...],
        strengths: tuple[float, ...], gamma: float,
        zero_threshold_multiplier: float = 0.0,
        background_low_frequency_strength: float = 0.0,
        background_chroma_low_frequency_strength: float | None = None) -> None:
    """批量处理互不重叠的 8×8 块；仅改变预量化计算速度。"""
    block_rows, block_cols = padded.shape[0] // 8, padded.shape[1] // 8
    blocks = padded.reshape(block_rows, 8, block_cols, 8, 3).transpose(0, 2, 4, 1, 3)
    row_indices = np.arange(block_rows) * 8 // patch_size
    col_indices = np.arange(block_cols) * 8 // patch_size
    block_weights = weights[row_indices[:, None], col_indices[None, :]]
    attenuation = np.power(1.0 - block_weights.astype(np.float64), gamma)
    basis = _dct_basis()
    chroma_low = (background_low_frequency_strength
                  if background_chroma_low_frequency_strength is None
                  else background_chroma_low_frequency_strength)
    low_strengths = (background_low_frequency_strength, chroma_low, chroma_low)
    for channel, (table, strength) in enumerate(zip(tables, strengths)):
        low_strength = low_strengths[channel]
        if strength == 0 and low_strength == 0:
            continue
        multiplier = 1.0 + np.floor(
            attenuation[:, :, None, None] *
            (strength * frequency_weight[None, None, :, :] +
             low_strength *
             low_frequency_weight[None, None, :, :]) + 0.5)
        changed = multiplier > 1
        selected = np.any(changed, axis=(2, 3)) & (block_weights < 1)
        if not np.any(selected):
            continue
        channel_blocks = blocks[:, :, channel]
        coefficients = np.matmul(
            np.matmul(basis, channel_blocks[selected] - np.float32(128)), basis.T)
        step = table[None, :, :] * multiplier[selected]
        quantized = step * np.rint(coefficients / step)
        if zero_threshold_multiplier:
            # 论文描述对低重要性区域中可忽略的变换系数置零；仅在显式消融时启用。
            negligible = changed[selected] & (
                np.abs(coefficients) < zero_threshold_multiplier * step)
            quantized[negligible] = 0.0
        coefficients[changed[selected]] = quantized[changed[selected]]
        reconstructed = np.matmul(np.matmul(basis.T, coefficients), basis)
        channel_blocks[selected] = reconstructed + np.float32(128)


def _prequantize(image: Image.Image, importance: np.ndarray, patch_size: int,
                 quality: int, subsampling: int, cutoff: int,
                 luma_strength: float, chroma_strength: float, gamma: float,
                 importance_ratio_low: float, importance_ratio_high: float,
                 background_lowpass_sigma: float = 0.0,
                 fast_prequant: bool = False,
                 zero_threshold_multiplier: float = 0.0,
                 roi_halo_pixels: float = 0.0,
                 protected_patch_mask: np.ndarray | None = None,
                 background_low_frequency_strength: float = 0.0,
                 background_chroma_low_frequency_strength: float | None = None) -> Image.Image:
    if not isinstance(fast_prequant, bool):
        raise TypeError("fast_prequant 必须是布尔值")
    if not np.isfinite(zero_threshold_multiplier) or zero_threshold_multiplier < 0:
        raise ValueError("置零阈值倍率必须是有限非负数")
    weights = _patch_weights(importance, image, patch_size,
                             importance_ratio_low, importance_ratio_high)
    if protected_patch_mask is not None:
        protected = np.asarray(protected_patch_mask)
        if protected.shape != weights.shape or protected.dtype != np.bool_:
            raise ValueError("protected_patch_mask 必须是与重要性图同形的布尔网格")
        # 已被语义桶选中的 patch 不再受背景粗量化影响。
        weights = np.where(protected, np.float32(1), weights)
    image = _background_lowpass(image, weights, patch_size,
                                background_lowpass_sigma, roi_halo_pixels)
    chroma_low = (background_low_frequency_strength
                  if background_chroma_low_frequency_strength is None
                  else background_chroma_low_frequency_strength)
    low_strengths = (background_low_frequency_strength, chroma_low, chroma_low)
    all_low_zero = all(value == 0 for value in low_strengths)
    # 无差异的重要性或零强度直接走原图，避免无意义的颜色空间往返。
    if ((luma_strength == 0 and chroma_strength == 0 and
         all_low_zero) or
            (float(weights.max()) == float(weights.min()) and
             (all_low_zero or float(weights.max()) == 1))):
        return image
    luma_table, chroma_table = _codec_quant_tables(quality, subsampling)
    frequency = np.add.outer(np.arange(8), np.arange(8))
    frequency_weight = (np.maximum(frequency - cutoff, 0) / (14 - cutoff)
                        if cutoff < 14 else np.zeros((8, 8), dtype=np.float32))
    # 背景的 DC 与低频 AC 也可显式粗量化；1/(1+u+v) 保留低频层次，
    # 而受保护 patch 的权重为 1，其所有系数倍率始终为 1。
    low_frequency_weight = np.where(frequency <= cutoff, 1.0 / (1.0 + frequency), 0.0)
    # 保持 JPEG 的 Y、Cb、Cr 分量顺序；保存 YCbCr 图像时不会再次做 RGB->YCbCr 转换。
    ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.float32)
    height, width = ycbcr.shape[:2]
    padded = np.pad(ycbcr, ((0, (-height) % 8), (0, (-width) % 8), (0, 0)), mode="edge")
    tables = (luma_table, chroma_table, chroma_table)
    strengths = (luma_strength, chroma_strength, chroma_strength)
    # 同一个权重图可选择逐块实现或向量化实现；两者都只在背景侧放大
    # JPEG 表对应的量化步长，重要 patch 的步长倍率保持为 1。
    if fast_prequant:
        _prequantize_blocks_fast(padded, weights, patch_size, frequency_weight,
                                 low_frequency_weight, tables, strengths, gamma,
                                 zero_threshold_multiplier,
                                 background_low_frequency_strength,
                                 background_chroma_low_frequency_strength)
    else:
        for y in range(0, padded.shape[0], 8):
            for x in range(0, padded.shape[1], 8):
                weight = float(weights[y // patch_size, x // patch_size])
                if weight >= 1.0:
                    continue
                attenuation = (1.0 - weight) ** gamma
                for channel, (table, strength) in enumerate(zip(tables, strengths)):
                    low_strength = low_strengths[channel]
                    if strength == 0 and low_strength == 0:
                        continue
                    # 整数倍步长与 JPEG 原量化表对齐；前景系数倍率仍为 1。
                    multiplier = 1.0 + np.floor(
                        attenuation * (strength * frequency_weight +
                                       low_strength *
                                       low_frequency_weight) + 0.5)
                    changed = multiplier > 1
                    if not changed.any():
                        continue
                    step = table * multiplier
                    block = padded[y:y + 8, x:x + 8, channel] - 128.0
                    coefficients = cv2.dct(block)
                    quantized = step * np.rint(coefficients / step)
                    if zero_threshold_multiplier:
                        negligible = changed & (np.abs(coefficients) <
                                                 zero_threshold_multiplier * step)
                        quantized[negligible] = 0.0
                    coefficients[changed] = quantized[changed]
                    padded[y:y + 8, x:x + 8, channel] = cv2.idct(coefficients) + 128.0
    result = np.clip(np.rint(padded[:height, :width]), 0, 255).astype(np.uint8)
    return Image.fromarray(result, "YCbCr")


def _restore_protected_rgb(prepared: Image.Image, original: Image.Image,
                           protected_patch_mask: np.ndarray,
                           patch_size: int) -> Image.Image:
    """预量化后把已选区域的 RGB 原像素放回，再交给同一个标准 JPEG 编码器。"""
    selected = np.asarray(protected_patch_mask)
    pixels = np.repeat(np.repeat(selected, patch_size, axis=0),
                       patch_size, axis=1)[:original.height, :original.width]
    restored = np.array(prepared.convert("RGB"), dtype=np.uint8, copy=True)
    source = np.asarray(original.convert("RGB"), dtype=np.uint8)
    restored[pixels] = source[pixels]
    return Image.fromarray(restored, "RGB")


def encode_task_jpeg(image: Image.Image, importance: np.ndarray, patch_size: int,
                     quality: int, subsampling: int, cutoff: int,
                     *, luma_strength: float = 2.0, chroma_strength: float = 2.0,
                     gamma: float = 1.0, importance_ratio_low: float = 1.0,
                     importance_ratio_high: float = 4.0,
                     background_lowpass_sigma: float = 0.0,
                     fast_prequant: bool = False,
                     zero_threshold_multiplier: float = 0.0,
                     roi_halo_pixels: float = 0.0,
                     restart_marker_rows: int = 0,
                     restart_marker_blocks: int = 0,
                     optimize_huffman: bool = False,
                     protected_patch_mask: np.ndarray | None = None,
                     background_low_frequency_strength: float = 0.0,
                     background_chroma_low_frequency_strength: float | None = None) -> bytes:
    """重要性驱动预量化；返回任何标准 JPEG 解码器均可读取的码流。"""
    _validate_parameters(patch_size, quality, subsampling, cutoff, luma_strength,
                         chroma_strength, gamma, importance_ratio_low,
                         importance_ratio_high, background_low_frequency_strength,
                         background_chroma_low_frequency_strength)
    prepared = _prequantize(image, importance, patch_size, quality, subsampling, cutoff,
                            luma_strength, chroma_strength, gamma,
                            importance_ratio_low, importance_ratio_high,
                            background_lowpass_sigma, fast_prequant,
                            zero_threshold_multiplier, roi_halo_pixels,
                            protected_patch_mask, background_low_frequency_strength,
                            background_chroma_low_frequency_strength)
    if protected_patch_mask is not None:
        prepared = _restore_protected_rgb(prepared, image, protected_patch_mask,
                                          patch_size)
    if restart_marker_rows not in (0, 1):
        raise ValueError("当前传输实验仅支持 restart_marker_rows=1")
    if restart_marker_blocks < 0 or (restart_marker_blocks and restart_marker_rows):
        raise ValueError("JPEG restart MCU 间隔无效")
    return _jpeg_bytes(prepared, quality, subsampling, restart_marker_rows,
                       restart_marker_blocks, optimize_huffman)


def encode_under_budget(image: Image.Image, importance: np.ndarray, patch_size: int,
                        budget_bytes: int, subsampling: int, cutoff: int,
                        *, luma_strength: float = 2.0, chroma_strength: float = 2.0,
                        gamma: float = 1.0, importance_ratio_low: float = 1.0,
                        importance_ratio_high: float = 4.0,
                        background_lowpass_sigma: float = 0.0,
                        fast_prequant: bool = False,
                        zero_threshold_multiplier: float = 0.0,
                        roi_halo_pixels: float = 0.0,
                        allow_420: bool = False,
                        max_quality: int = 95, restart_marker_rows: int = 0,
                        packet_header_bytes: int = 0,
                        restart_marker_blocks: int = 0,
                        packet_manifest_bytes_per_interval: int = 0,
                        optimize_huffman: bool = False,
                        background_low_frequency_strength: float = 0.0,
                        background_chroma_low_frequency_strength: float | None = None) -> tuple[bytes, dict]:
    """按真实 JPEG 码流和可选的每包传输头选质量。"""
    if budget_bytes <= 0:
        raise ValueError("码流预算必须为正")
    _validate_parameters(patch_size, max_quality, subsampling, cutoff, luma_strength,
                         chroma_strength, gamma, importance_ratio_low,
                         importance_ratio_high, background_low_frequency_strength,
                         background_chroma_low_frequency_strength)
    if packet_header_bytes < 0 or packet_manifest_bytes_per_interval < 0:
        raise ValueError("传输开销不能为负")
    if (packet_header_bytes or packet_manifest_bytes_per_interval) and not (
            restart_marker_rows or restart_marker_blocks):
        raise ValueError("按包计预算时必须设置 JPEG restart marker")
    if (restart_marker_rows or restart_marker_blocks) and not (
            subsampling == 0 or subsampling == 2 and allow_420):
        raise ValueError("按 MCU 区间分包要求 JPEG 4:4:4 (subsampling=0)")
    cache: dict[int, bytes] = {}

    def wire_size(payload: bytes) -> int:
        if not packet_header_bytes and not packet_manifest_bytes_per_interval:
            return len(payload)
        from replay.transmission.packet_loss import (count_entropy_intervals,
                                                     count_entropy_packets)
        return (len(payload) + packet_header_bytes * count_entropy_packets(
            payload, allow_420=allow_420)
                + packet_manifest_bytes_per_interval * count_entropy_intervals(
                    payload, allow_420=allow_420))

    def encoded(quality: int) -> bytes:
        if quality not in cache:
            cache[quality] = encode_task_jpeg(
                image, importance, patch_size, quality, subsampling, cutoff,
                luma_strength=luma_strength, chroma_strength=chroma_strength,
                gamma=gamma, importance_ratio_low=importance_ratio_low,
                importance_ratio_high=importance_ratio_high,
                background_lowpass_sigma=background_lowpass_sigma,
                fast_prequant=fast_prequant,
                zero_threshold_multiplier=zero_threshold_multiplier,
                background_low_frequency_strength=background_low_frequency_strength,
                background_chroma_low_frequency_strength=background_chroma_low_frequency_strength,
                roi_halo_pixels=roi_halo_pixels,
                restart_marker_rows=restart_marker_rows,
                restart_marker_blocks=restart_marker_blocks,
                optimize_huffman=optimize_huffman)
        return cache[quality]

    # 预算按实际压缩后的码流（可选加上包头/清单）核算，不用质量档位估算字节数。
    minimum = encoded(1)
    if wire_size(minimum) > budget_bytes:
        raise ValueError(f"原尺寸下无法满足 {budget_bytes} 字节预算；quality=1 需 {wire_size(minimum)} 字节")
    best: tuple[bytes, int] = minimum, 1
    low, high = 2, max_quality
    while low <= high:
        quality = (low + high) // 2
        payload = encoded(quality)
        if wire_size(payload) <= budget_bytes:
            best = payload, quality
            low = quality + 1
        else:
            high = quality - 1
    payload, quality = best
    # 预量化使大小不一定严格单调，补查相邻质量以实际码流为准。
    for candidate in range(quality + 1, min(max_quality, quality + 3) + 1):
        next_payload = encoded(candidate)
        if wire_size(next_payload) <= budget_bytes:
            payload, quality = next_payload, candidate
    return payload, {"jpeg_bytes": len(payload), "jpeg_quality": quality,
                     "codec_wire_bytes": wire_size(payload),
                     "packet_header_bytes": packet_header_bytes,
                     "packet_manifest_bytes_per_interval": packet_manifest_bytes_per_interval,
                     "restart_marker_rows": restart_marker_rows,
                     "restart_marker_blocks": restart_marker_blocks,
                     "optimize_huffman": optimize_huffman,
                     "luma_strength": luma_strength, "chroma_strength": chroma_strength,
                     "gamma": gamma, "importance_ratio_low": importance_ratio_low,
                     "importance_ratio_high": importance_ratio_high,
                     "background_lowpass_sigma": background_lowpass_sigma,
                     "zero_threshold_multiplier": zero_threshold_multiplier,
                     "background_low_frequency_strength": background_low_frequency_strength,
                     "background_chroma_low_frequency_strength": (
                         background_low_frequency_strength
                         if background_chroma_low_frequency_strength is None
                         else background_chroma_low_frequency_strength),
                     "roi_halo_pixels": roi_halo_pixels,
                     "allow_420": allow_420,
                     "fast_prequant": fast_prequant,
                     "budget_bytes": budget_bytes}


def decode_jpeg(payload: bytes) -> Image.Image:
    with Image.open(BytesIO(payload)) as image:
        return image.convert("RGB")
