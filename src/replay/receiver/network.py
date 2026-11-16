"""FiLM 条件化 UNet；输入输出始终保持原图宽高。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_dim: int, conditioned: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        groups = 8 if out_channels % 8 == 0 else (4 if out_channels % 4 == 0 else 2)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.film = nn.Linear(context_dim, 2 * out_channels) if conditioned else None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.norm1(self.conv1(x)))
        if self.film is not None:
            scale, shift = self.film(context).chunk(2, dim=-1)
            x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.silu(self.norm2(self.conv2(x)))


class FiLMUNet(nn.Module):
    def __init__(self, base_channels: int = 50, context_dim: int = 1024):
        super().__init__()
        c = base_channels
        self.context_mlp = nn.Sequential(nn.Linear(context_dim, 192), nn.SiLU(),
                                         nn.Linear(192, context_dim))
        self.down1 = Block(4, c, context_dim)
        self.down2 = Block(c, c * 2, context_dim)
        self.down3 = Block(c * 2, c * 4, context_dim)
        self.down4 = Block(c * 4, c * 8, context_dim)
        self.middle = Block(c * 8, c * 8, context_dim)
        # 论文 §3.3 明确 FiLM 作用在解码器各阶段，而不是编码器。
        self.up4 = Block(c * 16, c * 4, context_dim, conditioned=True)
        self.up3 = Block(c * 8, c * 2, context_dim, conditioned=True)
        self.up2 = Block(c * 4, c, context_dim, conditioned=True)
        self.up1 = Block(c * 2, c, context_dim, conditioned=True)
        self.out = nn.Conv2d(c, 3, 1)

    def forward(self, image: torch.Tensor, missing: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        context = context + self.context_mlp(context)
        # RGB 三通道加一通道缺失掩码；补边只为四次 2 倍下采样对齐尺寸。
        x = F.pad(torch.cat([image, missing], dim=1), (0, (-width) % 16, 0, (-height) % 16))
        a = self.down1(x, context)
        b = self.down2(F.avg_pool2d(a, 2), context)
        c = self.down3(F.avg_pool2d(b, 2), context)
        d = self.down4(F.avg_pool2d(c, 2), context)
        x = self.middle(F.avg_pool2d(d, 2), context)
        # 跳跃连接传递可见区域的细节；每一级解码块再由检索上下文调制。
        for skip, layer in ((d, self.up4), (c, self.up3), (b, self.up2), (a, self.up1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, skip], dim=1), context)
        # 输出恢复后的 RGB；只裁掉为 UNet 步长补上的像素。
        return self.out(x)[:, :, :height, :width].sigmoid()
