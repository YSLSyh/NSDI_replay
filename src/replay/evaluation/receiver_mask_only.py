"""接收端掩码限定输出消融：仅替换 JPEG 重启区间中确定丢失的像素。"""

from __future__ import annotations

import numpy as np
from PIL import Image


def compose_mask_only(decoded: Image.Image, restored: Image.Image,
                      missing_pixels: np.ndarray) -> Image.Image:
    """保留解码图的幸存区域，缺失区采用网络输出；不扩张未经验证的掩码。"""
    if decoded.size != restored.size:
        raise ValueError("直接解码图与接收端输出的尺寸不一致")
    mask = np.asarray(missing_pixels)
    if (mask.ndim != 2 or mask.shape != (decoded.height, decoded.width)
            or not np.issubdtype(mask.dtype, np.number)
            or not np.isfinite(mask).all()
            or not np.logical_or(mask == 0, mask == 1).all()):
        raise ValueError("缺失图必须为与图像同尺寸的二值 0/1 掩码")
    source = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    prediction = np.asarray(restored.convert("RGB"), dtype=np.uint8)
    # 原始像素以整数原样保留；避免全图生成器改坏未丢失的车、人或路面。
    combined = source.copy()
    combined[mask == 1] = prediction[mask == 1]
    return Image.fromarray(combined, "RGB")
