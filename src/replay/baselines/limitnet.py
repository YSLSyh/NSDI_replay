"""LimitNet 原生分辨率适配与可解码的渐进式潜变量码流。

网络层与优先级公式取自 Kiel University 的 MIT 许可 LimitNet 仓库；
逐层位掩码、zlib 封包和逐图量化范围是本复现为真实字节计数新增的适配。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


MAGIC = b"RLIM1"
HEADER = struct.Struct(">5sIIIIH")
LAYER_HEADER = struct.Struct(">II")
CHANNELS = 12
RANGES_BYTES = CHANNELS * 2 * 4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Encoder(nn.Module):
    """官方 LimitNet 的三层卷积编码器，不固定输入宽高。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=4, padding=1)
        self.conv3 = nn.Conv2d(16, CHANNELS, 3, stride=1, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(image))))))


class Decoder(nn.Module):
    """官方 LimitNet 的反卷积解码器。"""

    def __init__(self) -> None:
        super().__init__()
        self.t_conv1 = nn.ConvTranspose2d(CHANNELS, 64, 7, stride=2, padding=3,
                                          output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 64, 5, stride=4, padding=1,
                                          output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.t_conv5 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.t_conv1(latent))
        value = F.relu(self.t_conv2(value))
        value = F.relu(self.t_conv3(value))
        value = F.relu(self.t_conv4(value))
        return torch.sigmoid(self.t_conv5(value))


class SalDecoder(nn.Module):
    """官方 LimitNet 的显著性分支。"""

    def __init__(self) -> None:
        super().__init__()
        self.sal_t_conv1 = nn.Conv2d(CHANNELS, 16, 3, stride=1, padding=1)
        self.sal_t_conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.sal_t_conv3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.sal_t_conv4 = nn.Conv2d(8, 4, 3, stride=1, padding=1)
        self.sal_t_conv5 = nn.Conv2d(4, 1, 3, stride=1, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.sal_t_conv1(latent))
        value = F.relu(self.sal_t_conv2(value))
        value = F.relu(self.sal_t_conv3(value))
        value = F.relu(self.sal_t_conv4(value))
        return torch.sigmoid(self.sal_t_conv5(value))


class LimitNetCore(nn.Module):
    """只保留编码、显著性、解码路径；官方分类器由统一下游模型替代。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.sal_decoder = SalDecoder()
        self.decoder = Decoder()

    def forward(self, normalized: torch.Tensor, keep_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(normalized)
        saliency = self.sal_decoder(latent)
        keep = priority_mask(saliency, keep_fraction)
        return self.decoder(latent * keep), saliency


def priority_map(saliency: torch.Tensor) -> torch.Tensor:
    """沿用官方 saliency + (12-channel_index)/5 的发送优先级。"""
    if saliency.ndim != 4 or saliency.shape[1] != 1:
        raise ValueError("显著性图形状应为 B×1×H×W")
    bias = torch.arange(CHANNELS, 0, -1, device=saliency.device,
                        dtype=saliency.dtype).view(1, CHANNELS, 1, 1) / 5.0
    return saliency + bias


def quantize_saliency(saliency: torch.Tensor) -> torch.Tensor:
    """复刻官方 demo 对显著性图采用的 8/255 量化步长。"""
    minimum = saliency.amin(dim=(1, 2, 3), keepdim=True)
    maximum = saliency.amax(dim=(1, 2, 3), keepdim=True)
    span = maximum - minimum
    normalized = (saliency - minimum) / span.clamp_min(1e-8)
    discretized = torch.floor(normalized.clamp(0, 1) * 255 / 8) * 8 / 255
    return discretized * span + minimum


def priority_mask(saliency: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    """把官方固定 12×28×28 阈值选择推广到任意潜变量宽高。"""
    if not 0.0 <= keep_fraction <= 1.0 or not math.isfinite(keep_fraction):
        raise ValueError("LimitNet 保留比例必须在 [0,1]")
    priority = priority_map(saliency)
    count = priority[0].numel()
    keep_count = min(count, max(0, int(round(count * keep_fraction))))
    if keep_count == 0:
        return torch.zeros_like(priority)
    if keep_count == count:
        return torch.ones_like(priority)
    # 官方使用分位数；topk 精确保留给定数量，避免同值时比例漂移。
    flat = priority.flatten(start_dim=1)
    indices = flat.topk(keep_count, dim=1, sorted=False).indices
    mask = torch.zeros_like(flat).scatter_(1, indices, 1.0)
    return mask.reshape_as(priority)


@dataclass(frozen=True)
class LimitNetStream:
    """每层都是独立可校验封包；任意完整前缀都能解码。"""

    header: bytes
    layers: tuple[bytes, ...]

    def to_bytes(self, received_layers: int | None = None) -> bytes:
        count = len(self.layers) if received_layers is None else received_layers
        if not 0 <= count <= len(self.layers):
            raise ValueError("接收层数越界")
        return self.header + b"".join(self.layers[:count])

    def prefix_sizes(self) -> list[int]:
        sizes = []
        total = len(self.header)
        for layer in self.layers:
            total += len(layer)
            sizes.append(total)
        return sizes


@dataclass(frozen=True)
class LimitNetResult:
    image: Image.Image
    total_bytes: int
    received_fraction: float
    layers_received: int
    total_layers: int
    source: str


def _image_tensor(image: Image.Image, device: torch.device) -> tuple[torch.Tensor, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = torch.from_numpy(np.asarray(rgb, dtype=np.uint8).copy()).permute(2, 0, 1)[None]
    pixels = pixels.to(device).float().div_(255)
    # 填充仅用于卷积网格对齐；解码后裁回原尺寸，原图像素不缩放。
    pixels = F.pad(pixels, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
    mean = pixels.new_tensor(MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(STD).view(1, 3, 1, 1)
    return (pixels - mean) / std, width, height


class LimitNetCodec:
    """官方网络权重 + 项目渐进式实字节传输适配。"""

    def __init__(self, model: LimitNetCore, device: torch.device,
                 *, layers: int = 20, source: str = "unknown") -> None:
        if not 1 <= layers <= 100:
            raise ValueError("渐进层数必须在 1 到 100 之间")
        self.model = model.to(device).eval()
        self.device = device
        self.layers = layers
        self.source = source

    @torch.inference_mode()
    def encode_full(self, image: Image.Image,
                    *, focus_pixels: np.ndarray | None = None) -> LimitNetStream:
        normalized, width, height = _image_tensor(image, self.device)
        latent = self.model.encoder(normalized)
        if focus_pixels is None:
            saliency = quantize_saliency(self.model.sal_decoder(latent))
        else:
            focus = np.asarray(focus_pixels, dtype=np.float32)
            if focus.shape != (height, width) or not np.isfinite(focus).all():
                raise ValueError("关注区域图必须是有限值 H×W，且与原图同尺寸")
            if focus.min() < 0 or focus.max() > 1:
                raise ValueError("关注区域图数值必须在 [0,1]")
            focus_tensor = torch.from_numpy(focus.copy())[None, None].to(self.device)
            focus_tensor = F.pad(focus_tensor, (0, (-width) % 8, 0, (-height) % 8))
            # 与 KDC 使用同一 GT 关注区域；仅替换官方显著性分支输出。
            saliency = F.interpolate(focus_tensor, size=latent.shape[-2:], mode="area")
        priority = priority_map(saliency)
        if latent.shape[0] != 1 or latent.shape[1] != CHANNELS:
            raise ValueError("LimitNet 编码器应输出 1×12×H×W")
        latent_np = latent[0].float().cpu().numpy()
        priority_np = priority[0].float().cpu().numpy()
        _, latent_height, latent_width = latent_np.shape
        minima = latent_np.min(axis=(1, 2)).astype("<f4")
        maxima = latent_np.max(axis=(1, 2)).astype("<f4")
        span = np.maximum(maxima - minima, 1e-8)
        # 官方 demo 的 8-bit/步长 4 量化，即 64 个有效量化等级。
        quantized = np.floor(np.clip((latent_np - minima[:, None, None]) /
                                     span[:, None, None], 0, 1) * 255 / 4).astype(np.uint8)
        order = np.argsort(-priority_np.reshape(-1), kind="stable")
        values = quantized.reshape(-1)
        total = values.size
        mask_bytes = (total + 7) // 8
        chunk = (total + self.layers - 1) // self.layers
        header = (HEADER.pack(MAGIC, width, height, latent_width, latent_height,
                              self.layers) + minima.tobytes() + maxima.tobytes())
        packets: list[bytes] = []
        for start in range(0, total, chunk):
            indices = np.sort(order[start:start + chunk])
            mask = np.zeros(total, dtype=np.uint8)
            mask[indices] = 1
            packed_mask = np.packbits(mask, bitorder="little").tobytes()
            assert len(packed_mask) == mask_bytes
            raw = packed_mask + values[indices].tobytes()
            compressed = zlib.compress(raw, level=6)
            packets.append(LAYER_HEADER.pack(len(compressed), len(indices)) + compressed)
        return LimitNetStream(header, tuple(packets))

    @torch.inference_mode()
    def decode_bytes(self, payload: bytes) -> Image.Image:
        if len(payload) < HEADER.size + RANGES_BYTES:
            raise ValueError("LimitNet 码流头不完整")
        magic, width, height, latent_width, latent_height, layers = HEADER.unpack_from(payload)
        if magic != MAGIC or not (1 <= layers <= 100):
            raise ValueError("无效的 LimitNet 码流版本")
        total = CHANNELS * latent_width * latent_height
        if total <= 0 or total > 20_000_000:
            raise ValueError("LimitNet 潜变量尺寸无效")
        offset = HEADER.size
        minima = np.frombuffer(payload, dtype="<f4", count=CHANNELS, offset=offset)
        offset += CHANNELS * 4
        maxima = np.frombuffer(payload, dtype="<f4", count=CHANNELS, offset=offset)
        offset += CHANNELS * 4
        if not np.isfinite(minima).all() or not np.isfinite(maxima).all():
            raise ValueError("LimitNet 量化范围无效")
        q_flat = np.zeros(total, dtype=np.uint8)
        received = np.zeros(total, dtype=bool)
        mask_bytes = (total + 7) // 8
        packets = 0
        while offset < len(payload):
            if packets >= layers or offset + LAYER_HEADER.size > len(payload):
                raise ValueError("LimitNet 渐进层头不完整")
            compressed_size, count = LAYER_HEADER.unpack_from(payload, offset)
            offset += LAYER_HEADER.size
            if offset + compressed_size > len(payload):
                raise ValueError("LimitNet 渐进层数据不完整")
            raw = zlib.decompress(payload[offset:offset + compressed_size])
            offset += compressed_size
            if len(raw) != mask_bytes + count:
                raise ValueError("LimitNet 渐进层长度与掩码不符")
            selected = np.unpackbits(np.frombuffer(raw[:mask_bytes], dtype=np.uint8),
                                     bitorder="little")[:total].astype(bool)
            if int(selected.sum()) != count or np.any(received & selected):
                raise ValueError("LimitNet 渐进层存在重复或错误位置")
            q_flat[selected] = np.frombuffer(raw[mask_bytes:], dtype=np.uint8)
            received |= selected
            packets += 1
        channel = np.arange(total, dtype=np.int64) // (latent_width * latent_height)
        values = (q_flat.astype(np.float32) * 4 / 255 *
                  (maxima - minima)[channel] + minima[channel])
        values[~received] = 0.0
        latent = torch.from_numpy(values.reshape(1, CHANNELS, latent_height,
                                                  latent_width).copy()).to(self.device)
        output = self.model.decoder(latent)[0, :, :height, :width]
        pixels = (output.permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255)
        return Image.fromarray(pixels.astype(np.uint8), "RGB")

    def decode_prefix(self, stream: LimitNetStream, received_layers: int) -> Image.Image:
        return self.decode_bytes(stream.to_bytes(received_layers))

    def encode_decode(self, image: Image.Image, *, target_bytes: int | None = None,
                      missing_pixels: np.ndarray | None = None,
                      focus_pixels: np.ndarray | None = None) -> LimitNetResult:
        """按真实封包长度选最大可接收前缀；缺失图与其他方法共用。"""
        stream = self.encode_full(image, focus_pixels=focus_pixels)
        sizes = stream.prefix_sizes()
        if target_bytes is None:
            count = len(sizes)
        else:
            count = sum(size <= target_bytes for size in sizes)
            if target_bytes < len(stream.header):
                raise ValueError("字节预算不足以发送 LimitNet 码流头")
        encoded = stream.to_bytes(count)
        restored = self.decode_bytes(encoded)
        if missing_pixels is not None:
            mask = np.asarray(missing_pixels, dtype=np.float32)
            if mask.shape != (restored.height, restored.width):
                raise ValueError("通道缺失图与重建图尺寸不一致")
            pixels = np.asarray(restored, dtype=np.uint8).copy()
            pixels[mask >= 0.5] = 0
            restored = Image.fromarray(pixels, "RGB")
        source = self.source + ("+oracle_roi" if focus_pixels is not None else "")
        return LimitNetResult(restored, len(encoded), count / len(stream.layers),
                              count, len(stream.layers), source)


def load_limitnet(checkpoint: Path, device: torch.device,
                  *, layers: int = 20) -> LimitNetCodec:
    """读取安全的纯 state_dict；官方 pickle 需先用转换脚本显式转换。"""
    value = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("LimitNet 权重须为纯 state_dict 或包含 state_dict 的字典")
    state = value.get("state_dict", value)
    if not isinstance(state, dict):
        raise ValueError("LimitNet checkpoint 缺少 state_dict")
    # 官方完整模型还含分类器；统一分割头由外部 SegFormer-B5 提供。
    core = LimitNetCore()
    core.load_state_dict({key: tensor for key, tensor in state.items()
                          if key.startswith(("encoder.", "sal_decoder.", "decoder."))},
                         strict=True)
    metadata = value.get("metadata", {})
    source = str(metadata.get("source", "pretrained_only")) if isinstance(metadata, dict) else "unknown"
    return LimitNetCodec(core, device, layers=layers, source=source)
