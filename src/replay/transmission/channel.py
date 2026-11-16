"""图像模态的应用层语义 token 丢失代理；不是实际 5G 轨迹或字节包丢失。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from PIL import Image


# 论文未公开逐帧无线轨迹；这些范围是可复现实验的项目代理条件。
_PROFILES = {
    "highway": {"throughput_mbps": (5.0, 35.0),
                "loss_rates": {"low": 0.02, "medium": 0.05, "high": 0.08}},
    "scooter": {"throughput_mbps": (10.0, 31.0),
                "loss_rates": {"low": 0.03, "medium": 0.075, "high": 0.12}},
}


@dataclass(frozen=True)
class ChannelOutcome:
    """damaged 是解码后图像；missing_pixels 中 1 表示接收端应恢复的区域。"""

    damaged: Image.Image
    missing_pixels: np.ndarray
    lost_patches: np.ndarray
    target_loss_rate: float
    actual_loss_rate: float
    lost_patch_count: int
    total_patch_count: int
    throughput_mbps: float
    transmitted_bytes: int
    fixed_overhead_ms: float
    transfer_latency_ms: float
    proxy_kind: str = "post_jpeg_decode_semantic_token_erasure"


def _seed(seed: int, image_id: str, scenario: str, purpose: str) -> int:
    # 不使用 Python hash()：其随机化会使不同进程得到不同缺失图。
    payload = json.dumps([seed, image_id, scenario, purpose],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def simulate_token_erasure(
    decoded_image: Image.Image,
    *,
    image_id: str,
    scenario: str,
    level: str,
    seed: int,
    transmitted_bytes: int,
    patch_size: int = 16,
    fixed_overhead_ms: float = 10.0,
) -> ChannelOutcome:
    """对 JPEG 解码图做块丢失；同一图像条件对各 codec 共享缺失位置。

    该代理在 codec 解码后运行，不能据此声称模拟了 JPEG 字节包损坏、重传或
    论文没有公开的 5G 无线轨迹。传输时间只按实际发送字节和代理吞吐率计算。
    """
    if scenario not in _PROFILES:
        raise ValueError(f"未知信道场景：{scenario}")
    profile = _PROFILES[scenario]
    if level not in profile["loss_rates"]:
        raise ValueError(f"未知丢失级别：{level}")
    if not image_id or not isinstance(image_id, str):
        raise ValueError("image_id 必须是非空字符串")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")
    if not isinstance(transmitted_bytes, int) or transmitted_bytes < 0:
        raise ValueError("transmitted_bytes 必须是非负整数")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size 必须是正整数")
    if not math.isfinite(fixed_overhead_ms) or fixed_overhead_ms < 0:
        raise ValueError("fixed_overhead_ms 必须是有限非负数")
    width, height = decoded_image.size
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须为正")

    rows = (height + patch_size - 1) // patch_size
    cols = (width + patch_size - 1) // patch_size
    count = rows * cols
    rate = float(profile["loss_rates"][level])
    lost_count = min(count, int(math.floor(count * rate + 0.5)))
    # 各级别共用同一随机排列，使 high 的丢失集合包含 medium 和 low。
    order = np.random.default_rng(_seed(seed, image_id, scenario, "patch_order")).permutation(count)
    lost = np.zeros(count, dtype=bool)
    lost[order[:lost_count]] = True
    lost = lost.reshape(rows, cols)
    missing = np.repeat(np.repeat(lost, patch_size, axis=0),
                        patch_size, axis=1)[:height, :width]
    pixels = np.asarray(decoded_image.convert("RGB"), dtype=np.uint8).copy()
    pixels[missing] = 0
    damaged = Image.fromarray(pixels, "RGB")

    low, high = profile["throughput_mbps"]
    throughput = float(np.random.default_rng(
        _seed(seed, image_id, scenario, "throughput")
    ).uniform(low, high))
    latency = fixed_overhead_ms + transmitted_bytes * 8 / (throughput * 1_000_000) * 1_000
    return ChannelOutcome(
        damaged=damaged,
        missing_pixels=missing.astype(np.float32),
        lost_patches=lost,
        target_loss_rate=rate,
        actual_loss_rate=lost_count / count,
        lost_patch_count=lost_count,
        total_patch_count=count,
        throughput_mbps=throughput,
        transmitted_bytes=transmitted_bytes,
        fixed_overhead_ms=fixed_overhead_ms,
        transfer_latency_ms=latency,
    )
