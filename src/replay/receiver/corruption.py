"""以 ViT patch 为单位模拟论文中的 5–15% token 缺失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def drop_patches(images: torch.Tensor, patch: int, low: float, high: float) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("输入应为 BCHW RGB 图像")
    if not 0 <= low <= high <= 1:
        raise ValueError("丢失率范围无效")
    batch, _, height, width = images.shape
    rows = (height + patch - 1) // patch
    cols = (width + patch - 1) // patch
    rate = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(low, high)
    missing = torch.rand(batch, 1, rows, cols, device=images.device) < rate
    missing = F.interpolate(missing.float(), size=(rows * patch, cols * patch), mode="nearest")[:, :, :height, :width]
    # 缺失 token 以零值替代；额外的 mask 通道交给接收端区分真实黑色与缺失。
    return images * (1 - missing), missing


def drop_semantic_tokens(images: torch.Tensor, tokens: torch.Tensor, patch: int,
                         low: float, high: float,
                         generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """先对 CLIP 语义 token 网格丢失，再映射到接收端可见的缺失图像区域。"""
    if images.ndim != 4 or tokens.ndim != 4 or images.shape[0] != tokens.shape[0]:
        raise ValueError("images 应为 BCHW，tokens 应为 B×patch行×patch列×512")
    batch, _, height, width = images.shape
    rows, cols = (height + patch - 1) // patch, (width + patch - 1) // patch
    if tokens.shape[1:3] != (rows, cols):
        raise ValueError("token 网格必须逐一对应原图的 16×16 patch")
    if not 0 <= low <= high <= 1:
        raise ValueError("语义 token 丢失率范围无效")
    rates = low + (high - low) * torch.rand(batch, 1, 1, 1, device=images.device,
                                           generator=generator)
    lost = torch.rand(batch, rows, cols, 1, device=images.device, generator=generator) < rates
    surviving = tokens.to(images.device).float().masked_fill(lost, 0.0)
    counts = (~lost).sum(dim=(1, 2)).clamp_min(1)
    current = F.normalize(surviving.sum(dim=(1, 2)) / counts, dim=-1)
    missing = F.interpolate(lost.permute(0, 3, 1, 2).float(),
                            size=(rows * patch, cols * patch), mode="nearest")[:, :, :height, :width]
    # 仅用空间 mask 表示这些语义 token 对应的可见内容缺失；幸存 token 用于 SharedKB 上下文。
    return images * (1 - missing), missing, current
