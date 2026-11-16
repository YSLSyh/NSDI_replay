"""无 RST 普通 JPEG 的扫描熵分片丢失，沿用首轮合成 GE 随机事件。

JPEG 结构标记作为可靠控制信息传送并计入字节数；只有扫描熵数据承受
应用层分片丢失。此模块不代表实际 UDP、5G 轨迹或任何论文未公开的链路。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from collections.abc import Sequence

from PIL import Image

from replay.transmission.channel import _PROFILES
from replay.transmission.packet_loss import _packet_loss_events


@dataclass(frozen=True)
class PlainJpegLossOutcome:
    received_payload: bytes
    decoded: Image.Image | None
    decode_error: str | None
    packet_count: int
    lost_packet_indices: tuple[int, ...]
    target_loss_rate: float
    actual_loss_rate: float
    sent_bytes: int
    received_bytes: int
    reliable_control_bytes: int
    entropy_payload_bytes: int
    lost_entropy_bytes: int
    loss_model: str
    channel_kind: str = "synthetic_ge_entropy_fragment_loss_on_full_jpeg_no_rst"

    @property
    def lost_packets(self) -> int:
        return len(self.lost_packet_indices)

    @property
    def status(self) -> str:
        return "ok" if self.decoded is not None else "undecodable"


def split_plain_jpeg_scan(payload: bytes) -> tuple[bytes, bytes, bytes]:
    """返回可靠前缀、单一扫描熵数据、可靠 EOI；拒绝渐进式和 RST。"""
    if not isinstance(payload, bytes) or not payload.startswith(b"\xff\xd8"):
        raise ValueError("普通 JPEG 缺少 SOI")
    cursor = 2
    while cursor < len(payload):
        if payload[cursor] != 0xFF:
            raise ValueError("普通 JPEG 控制标记边界无效")
        while cursor < len(payload) and payload[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(payload):
            raise ValueError("普通 JPEG 标记截断")
        marker = payload[cursor]
        cursor += 1
        if marker == 0xD9:
            raise ValueError("普通 JPEG 缺少 SOS/熵数据")
        if marker == 0x01:
            continue
        if marker in range(0xD0, 0xD9) or marker == 0x00:
            raise ValueError("普通 JPEG 控制区包含非法 RST 或转义标记")
        if cursor + 2 > len(payload):
            raise ValueError("普通 JPEG 段长度截断")
        length = int.from_bytes(payload[cursor:cursor + 2], "big")
        if length < 2 or cursor + length > len(payload):
            raise ValueError("普通 JPEG 段长度无效")
        cursor += length
        if marker != 0xDA:
            continue
        scan_start = cursor
        while cursor < len(payload):
            if payload[cursor] != 0xFF:
                cursor += 1
                continue
            marker_start = cursor
            while cursor < len(payload) and payload[cursor] == 0xFF:
                cursor += 1
            if cursor >= len(payload):
                raise ValueError("普通 JPEG 扫描数据截断")
            code = payload[cursor]
            cursor += 1
            if code == 0x00:
                continue  # FF00 是熵字节转义，不是控制标记。
            if code != 0xD9 or cursor != len(payload):
                raise ValueError("仅支持单一扫描、无 RST 的普通 JPEG")
            if marker_start == scan_start:
                raise ValueError("普通 JPEG 熵流为空")
            return payload[:scan_start], payload[scan_start:marker_start], payload[marker_start:]
    raise ValueError("普通 JPEG 缺少 EOI")


def plain_jpeg_wire_bytes(payload: bytes, *, packet_payload_bytes: int = 1200,
                          packet_header_bytes: int = 8) -> int:
    """包括普通 JPEG 全部字节与扫描熵分片的逐包头。"""
    if (type(packet_payload_bytes) is not int or packet_payload_bytes <= 0 or
            type(packet_header_bytes) is not int or packet_header_bytes < 0):
        raise ValueError("分片字节数或包头无效")
    _, entropy, _ = split_plain_jpeg_scan(payload)
    return len(payload) + math.ceil(len(entropy) / packet_payload_bytes) * packet_header_bytes


def simulate_plain_jpeg_entropy_loss(
    payload: bytes, *, image_id: str, expected_size: tuple[int, int] | None = None,
    scenario: str = "scooter", level: str = "medium", seed: int = 2026,
    packet_payload_bytes: int = 1200, packet_header_bytes: int = 8,
    mean_burst_packets: float = 3.0,
    loss_trace: Sequence[bool] | None = None,
) -> PlainJpegLossOutcome:
    """删除确实丢失的熵分片，再用 Pillow 尝试真实 JPEG 解码。"""
    if scenario not in _PROFILES or level not in _PROFILES[scenario]["loss_rates"]:
        raise ValueError("未知的随机丢包场景或级别")
    if not image_id or not isinstance(image_id, str):
        raise ValueError("image_id 必须是非空字符串")
    if type(seed) is not int or type(packet_header_bytes) is not int or packet_header_bytes < 0:
        raise ValueError("seed 或包头字节数无效")
    if type(packet_payload_bytes) is not int or packet_payload_bytes <= 0:
        raise ValueError("packet_payload_bytes 必须为正整数")
    prefix, entropy, suffix = split_plain_jpeg_scan(payload)
    spans = tuple((start, min(start + packet_payload_bytes, len(entropy)))
                  for start in range(0, len(entropy), packet_payload_bytes))
    rate = float(_PROFILES[scenario]["loss_rates"][level])
    events, model = _packet_loss_events(
        len(spans), rate, seed, image_id, scenario,
        "gilbert_elliott", loss_trace, mean_burst_packets)
    lost = tuple(i for i, event in enumerate(events) if event)
    lost_bytes = sum(end - start for i, (start, end) in enumerate(spans) if events[i])
    received_entropy = b"".join(entropy[start:end] for i, (start, end) in enumerate(spans)
                                if not events[i])
    received = prefix + received_entropy + suffix
    decoded = None
    error = None
    try:
        with Image.open(BytesIO(received)) as candidate:
            candidate.load()
            if expected_size is not None and candidate.size != expected_size:
                raise ValueError("损坏 JPEG 解码后的尺寸不一致")
            decoded = candidate.convert("RGB")
    except (OSError, ValueError, SyntaxError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    sent_bytes = len(payload) + len(spans) * packet_header_bytes
    return PlainJpegLossOutcome(
        received_payload=received, decoded=decoded, decode_error=error,
        packet_count=len(spans), lost_packet_indices=lost,
        target_loss_rate=rate, actual_loss_rate=len(lost) / len(spans),
        sent_bytes=sent_bytes,
        received_bytes=sent_bytes - lost_bytes - len(lost) * packet_header_bytes,
        reliable_control_bytes=len(prefix) + len(suffix),
        entropy_payload_bytes=len(entropy), lost_entropy_bytes=lost_bytes,
        loss_model=model,
    )
