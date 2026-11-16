"""图 7 的普通 JPEG / Progressive JPEG 原生码流适配。

普通 JPEG 按预算搜索完整可解码文件；渐进式 JPEG 固定质量并按扫描
顺序交付可用前缀，保留两种基线各自的率失真策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import ceil
from typing import Sequence

from PIL import Image

from replay.transmission.baselines import encode_jpeg_baseline


@dataclass(frozen=True)
class JpegPrefixResult:
    image: Image.Image | None
    transmitted_codec_bytes: int
    transmitted_wire_bytes: int
    decoded_codec_bytes: int
    complete_scans: int
    full_stream: bool


class _JpegStream(bytes):
    """保留发送端已解析的扫描边界，供精确预算边界的仿真解码使用。"""

    def __new__(cls, payload: bytes, scan_ends: Sequence[int] = ()) -> _JpegStream:
        stream = super().__new__(cls, payload)
        stream.scan_ends = tuple(scan_ends)
        return stream


class JpegBudgetInfeasibleError(ValueError):
    """原尺寸 quality=1 JPEG 仍放不进预算。"""


def encode_jpeg(
    image: Image.Image, *, quality: int = 75, progressive: bool = False,
    subsampling: int = 0, optimize_huffman: bool = False,
) -> bytes:
    """原尺寸编码；默认 quality=75 对齐论文所报的 JPEG 基线配置。"""
    return encode_jpeg_baseline(
        image, quality, subsampling, progressive=progressive,
        optimize_huffman=optimize_huffman,
    )


def jpeg_wire_bytes(
    codec_bytes: int, *, packet_payload_bytes: int = 1200,
    packet_header_bytes: int = 0,
) -> int:
    """精确计入最后一个短包的包头；不把接收端补的 EOI 计入传输。"""
    if codec_bytes < 0 or packet_payload_bytes <= 0 or packet_header_bytes < 0:
        raise ValueError("JPEG 码流长度或分包参数无效")
    packet_count = (codec_bytes + packet_payload_bytes - 1) // packet_payload_bytes
    return codec_bytes + packet_count * packet_header_bytes


def _scan_end_offsets(payload: bytes) -> tuple[int, ...]:
    """解析 JPEG marker，返回每个完整 SOS 扫描的最后一个熵字节之后的位置。"""
    if not payload.startswith(b"\xff\xd8"):
        return ()
    ends: list[int] = []
    position = 2
    size = len(payload)
    while position < size:
        if payload[position] != 0xFF:
            return tuple(ends)
        while position < size and payload[position] == 0xFF:
            position += 1
        if position >= size:
            break
        marker = payload[position]
        position += 1
        if marker == 0xD9:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if marker == 0x00 or position + 2 > size:
            break
        segment_size = int.from_bytes(payload[position:position + 2], "big")
        if segment_size < 2 or position + segment_size > size:
            break
        position += segment_size
        if marker != 0xDA:
            continue
        # 在熵数据内 FF00 是转义字节，FFD0..FFD7 是重启标记。
        while position < size:
            if payload[position] != 0xFF:
                position += 1
                continue
            candidate = position
            next_position = position + 1
            while next_position < size and payload[next_position] == 0xFF:
                next_position += 1
            if next_position >= size:
                return tuple(ends)
            following = payload[next_position]
            if following == 0x00 or 0xD0 <= following <= 0xD7:
                position = next_position + 1
                continue
            ends.append(candidate)
            position = candidate
            break
        else:
            break
    return tuple(ends)


def _decode_rgb(payload: bytes) -> Image.Image | None:
    try:
        with Image.open(BytesIO(payload)) as loaded:
            loaded.load()
            return loaded.convert("RGB")
    except (OSError, ValueError):
        return None


def _decode_complete_progressive_prefix(
    payload: bytes, complete_scan_ends: Sequence[int] | None = None,
) -> tuple[Image.Image | None, int, int]:
    # 码流恰好截止于熵扫描末字节时，前缀本身没有下一个 marker 可供识别。
    # 发送端解析完整码流所得边界可判定该扫描确已结束；绝不使用越过前缀的扫描。
    ends = (_scan_end_offsets(payload) if complete_scan_ends is None else
            tuple(end for end in complete_scan_ends if 0 < end <= len(payload)))
    # 仅完整的扫描可用于提前解码；本地添加的 EOI 不作为传输字节。
    for end in reversed(ends):
        image = _decode_rgb(payload[:end] + b"\xff\xd9")
        if image is not None:
            return image, end, sum(scan_end <= end for scan_end in ends)
    return None, 0, 0


def _max_codec_prefix_bytes(
    payload_size: int, budget_bytes: int, *, packet_payload_bytes: int,
    packet_header_bytes: int,
) -> int:
    low, high = 0, payload_size
    while low < high:
        middle = (low + high + 1) // 2
        if jpeg_wire_bytes(middle, packet_payload_bytes=packet_payload_bytes,
                           packet_header_bytes=packet_header_bytes) <= budget_bytes:
            low = middle
        else:
            high = middle - 1
    return low


def deliver_jpeg_prefix(
    payload: bytes, budget_bytes: int, *, progressive: bool,
    packet_payload_bytes: int = 1200, packet_header_bytes: int = 0,
) -> JpegPrefixResult:
    """按总传输预算交付固定质量码流的可解码前缀。

    普通 JPEG 仅完整交付才可用；渐进式 JPEG 仅交付完整扫描。
    """
    if budget_bytes < 0:
        raise ValueError("预算不能为负")
    jpeg_wire_bytes(0, packet_payload_bytes=packet_payload_bytes,
                    packet_header_bytes=packet_header_bytes)
    full_wire = jpeg_wire_bytes(
        len(payload), packet_payload_bytes=packet_payload_bytes,
        packet_header_bytes=packet_header_bytes,
    )
    if full_wire <= budget_bytes:
        image = _decode_rgb(payload)
        return JpegPrefixResult(
            image=image, transmitted_codec_bytes=len(payload),
            transmitted_wire_bytes=full_wire,
            decoded_codec_bytes=len(payload) if image is not None else 0,
            complete_scans=len(_scan_end_offsets(payload)), full_stream=True,
        )
    prefix_bytes = _max_codec_prefix_bytes(
        len(payload), budget_bytes, packet_payload_bytes=packet_payload_bytes,
        packet_header_bytes=packet_header_bytes,
    )
    sent_wire = jpeg_wire_bytes(
        prefix_bytes, packet_payload_bytes=packet_payload_bytes,
        packet_header_bytes=packet_header_bytes,
    )
    if not progressive:
        return JpegPrefixResult(None, prefix_bytes, sent_wire, 0, 0, False)
    eligible = [end for end in _scan_end_offsets(payload) if end <= prefix_bytes]
    if not eligible:
        return JpegPrefixResult(None, prefix_bytes, sent_wire, 0, 0, False)
    # 某些 Pillow/libjpeg 版本不解某个早期扫描，回退到更早的完整扫描。
    for end in reversed(eligible):
        image = _decode_rgb(payload[:end] + b"\xff\xd9")
        if image is not None:
            return JpegPrefixResult(
                image, prefix_bytes, sent_wire,
                end, sum(scan_end <= end for scan_end in eligible), False,
            )
    return JpegPrefixResult(None, prefix_bytes, sent_wire, 0, 0, False)


def decode_received_jpeg(
    payload: bytes, *, loss_flags: Sequence[bool], progressive: bool,
    packet_payload_bytes: int = 1200,
    complete_scan_ends: Sequence[int] | None = None,
) -> Image.Image | None:
    """解码丢包后的原生 JPEG；渐进式仅利用首个丢包前的完整扫描。

    ``loss_flags[i]`` 为真表示第 i 个应用层数据包丢失。这样不会把
    丢失后续扫描的字节拼接成一个看似有效却被污染的 JPEG 码流。
    """
    if packet_payload_bytes <= 0:
        raise ValueError("每包载荷字节数必须为正")
    expected = ceil(len(payload) / packet_payload_bytes)
    if len(loss_flags) != expected:
        raise ValueError(f"丢包标志长度 {len(loss_flags)} 与数据包数 {expected} 不一致")
    first_loss = next((index for index, lost in enumerate(loss_flags) if lost), None)
    if first_loss is None:
        if payload.endswith(b"\xff\xd9"):
            return _decode_rgb(payload)
        if not progressive:
            return None
        image, _, _ = _decode_complete_progressive_prefix(payload, complete_scan_ends)
        return image
    if not progressive:
        return None
    prefix = payload[:first_loss * packet_payload_bytes]
    image, _, _ = _decode_complete_progressive_prefix(prefix, complete_scan_ends)
    return image


def search_jpeg_budget(
    image: Image.Image, budget_bytes: int, *, progressive: bool,
    subsampling: int = 0, max_quality: int = 95,
    packet_payload_bytes: int = 1200, packet_header_bytes: int = 0,
) -> tuple[bytes, dict]:
    """搜索预算内的完整 JPEG 文件；只对真实码流字节数作可行性判断。"""
    if budget_bytes <= 0 or not 1 <= max_quality <= 100:
        raise ValueError("JPEG 预算或质量范围无效")
    jpeg_wire_bytes(0, packet_payload_bytes=packet_payload_bytes,
                    packet_header_bytes=packet_header_bytes)
    cache: dict[int, bytes] = {}

    def encoded(quality: int) -> bytes:
        if quality not in cache:
            cache[quality] = encode_jpeg(
                image, quality=quality, progressive=progressive,
                subsampling=subsampling)
        return cache[quality]

    def fits(quality: int) -> bool:
        return jpeg_wire_bytes(
            len(encoded(quality)), packet_payload_bytes=packet_payload_bytes,
            packet_header_bytes=packet_header_bytes) <= budget_bytes

    minimum = encoded(1)
    minimum_wire = jpeg_wire_bytes(
        len(minimum), packet_payload_bytes=packet_payload_bytes,
        packet_header_bytes=packet_header_bytes)
    if minimum_wire > budget_bytes:
        raise JpegBudgetInfeasibleError(
            f"原尺寸 JPEG 质量 1 需 {minimum_wire} 字节，超过预算 {budget_bytes}")
    quality = 1
    low, high = 2, max_quality
    while low <= high:
        candidate = (low + high) // 2
        if fits(candidate):
            quality = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    # libjpeg 文件大小偶有不单调点，真实字节数再检查相邻质量。
    for candidate in range(quality + 1, min(max_quality, quality + 3) + 1):
        if fits(candidate):
            quality = candidate
    payload = encoded(quality)
    wire = jpeg_wire_bytes(
        len(payload), packet_payload_bytes=packet_payload_bytes,
        packet_header_bytes=packet_header_bytes,
    )
    return payload, {
        "codec_bytes": len(payload), "wire_bytes": wire,
        "jpeg_quality": quality, "progressive": progressive,
        "subsampling": subsampling, "packet_payload_bytes": packet_payload_bytes,
        "packet_header_bytes": packet_header_bytes,
        "original_size": [image.width, image.height],
        "budget_bytes": budget_bytes,
        "budget_policy": "complete_file_quality_search",
        "quality_search_policy": "binary_plus_three_neighbors",
    }


@dataclass(frozen=True)
class JpegClassificationCodec:
    """普通 JPEG 搜索完整文件；ProgJPEG 固定质量并传输扫描前缀。"""

    progressive: bool
    quality: int = 75
    subsampling: int = 0
    packet_payload_bytes: int = 1200
    packet_header_bytes: int = 0

    @property
    def name(self) -> str:
        return "prog_jpeg" if self.progressive else "jpeg"

    def encode(self, image: Image.Image, *, budget_bytes: int | None) -> tuple[bytes, dict]:
        full = encode_jpeg(
            image, quality=self.quality, progressive=self.progressive,
            subsampling=self.subsampling)
        full_wire = jpeg_wire_bytes(
            len(full), packet_payload_bytes=self.packet_payload_bytes,
            packet_header_bytes=self.packet_header_bytes)
        actual_quality: int | None = self.quality
        status = "ok"
        if budget_bytes is not None and budget_bytes < 0:
            raise ValueError("传输预算不能为负")
        if budget_bytes is None:
            sent = full
        elif self.progressive:
            prefix_size = _max_codec_prefix_bytes(
                len(full), budget_bytes,
                packet_payload_bytes=self.packet_payload_bytes,
                packet_header_bytes=self.packet_header_bytes)
            sent = full[:prefix_size]
        elif full_wire <= budget_bytes:
            sent = full
        else:
            try:
                sent, search_info = search_jpeg_budget(
                    image, budget_bytes, progressive=False,
                    subsampling=self.subsampling, max_quality=self.quality,
                    packet_payload_bytes=self.packet_payload_bytes,
                    packet_header_bytes=self.packet_header_bytes)
                actual_quality = search_info["jpeg_quality"]
            except JpegBudgetInfeasibleError:
                sent = b""
                actual_quality = None
                status = "infeasible_quality1"
        sent = _JpegStream(sent, _scan_end_offsets(full) if self.progressive else ())
        wire = jpeg_wire_bytes(
            len(sent), packet_payload_bytes=self.packet_payload_bytes,
            packet_header_bytes=self.packet_header_bytes)
        return sent, {
            "codec_bytes": len(sent), "wire_bytes": wire,
            "source_codec_bytes": len(full), "source_wire_bytes": full_wire,
            "jpeg_quality": actual_quality, "max_jpeg_quality": self.quality,
            "progressive": self.progressive,
            "subsampling": self.subsampling, "original_size": [image.width, image.height],
            "packet_payload_bytes": self.packet_payload_bytes,
            "packet_header_bytes": self.packet_header_bytes,
            "budget_bytes": budget_bytes,
            "budget_policy": ("fixed_quality_scan_prefix" if self.progressive else
                              "complete_file_quality_search"),
            "status": status,
        }

    def decode(
        self, payload: bytes, *, loss_flags: tuple[bool, ...],
        received_prefix_bytes: int | None,
    ) -> Image.Image | None:
        if received_prefix_bytes is None:
            prefix_size = len(payload)
        else:
            if not 0 <= received_prefix_bytes <= len(payload):
                raise ValueError("接收前缀长度超出 JPEG 码流")
            prefix_size = received_prefix_bytes
        prefix = payload[:prefix_size]
        prefix_packets = ceil(prefix_size / self.packet_payload_bytes)
        all_packets = ceil(len(payload) / self.packet_payload_bytes)
        if len(loss_flags) not in (prefix_packets, all_packets):
            raise ValueError("JPEG 丢包标志与发送码流长度不符")
        return decode_received_jpeg(
            prefix, loss_flags=loss_flags[:prefix_packets],
            progressive=self.progressive,
            packet_payload_bytes=self.packet_payload_bytes,
            complete_scan_ends=getattr(payload, "scan_ends", None))
