"""可部分接收的 JPEG 应用层分包线格式。

控制清单由可靠通道传送；熵数据包允许在应用层截止时间后缺失。控制清单
包含 JPEG 非熵结构和每个 restart interval 的位置，因此接收端不需要原图
或完整 JPEG 才能按丢包序号确定哪些区间不可解码。该封装是 replay 的设计，
原论文没有公开其字节级封装。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import struct
import zlib
from typing import Sequence


_MAGIC = b"RPLYAPP2"
_SCHEMA = 2
_SESSION_PRESENT = 1
# 长度依次为 session、frame、原 JPEG、restart 区间数、数据包数、载荷上限、包头长度。
_PREFIX = struct.Struct(">8sBBIIIIIII")
_INTERVAL = struct.Struct(">IIIIII")
_PACKET = struct.Struct(">II")  # 序号、载荷 CRC32；不足 packet_header_bytes 时补零。
_DIGEST_BYTES = 32
CONTROL_FIXED_BYTES = _PREFIX.size + _DIGEST_BYTES
INTERVAL_MANIFEST_BYTES = _INTERVAL.size
MIN_PACKET_HEADER_BYTES = _PACKET.size


@dataclass(frozen=True)
class PacketizedFrame:
    control: bytes
    packets: tuple[bytes, ...]

    @property
    def sent_bytes(self) -> int:
        return len(self.control) + sum(map(len, self.packets))


@dataclass(frozen=True)
class ReceivedFrame:
    session_header: bytes | None
    frame_header: bytes
    jpeg_payload: bytes
    lost_packet_indices: tuple[int, ...]
    lost_intervals: tuple[tuple[int, int], ...]
    lost_entropy_bytes: int
    receiver_discarded_bytes: int
    received_bytes: int


def _validate_plan(payload: bytes, intervals: Sequence[object], packets: Sequence[object],
                   packet_payload_bytes: int) -> None:
    if not intervals or not packets:
        raise ValueError("JPEG restart 区间和数据包不能为空")
    previous = 0
    for interval in intervals:
        start, end = int(interval.start), int(interval.end)
        if start < previous or end <= start or end > len(payload):
            raise ValueError("JPEG restart 区间必须有序、非空且互不重叠")
        previous = end
    position = int(intervals[0].start)
    interval_ordinal = 0
    for packet in packets:
        start, end = int(packet.start), int(packet.end)
        while interval_ordinal < len(intervals) and position == int(intervals[interval_ordinal].end):
            interval_ordinal += 1
            if interval_ordinal < len(intervals):
                position = int(intervals[interval_ordinal].start)
        if interval_ordinal >= len(intervals) or start != position or end <= start:
            raise ValueError("数据包未连续覆盖 JPEG 熵区间")
        if end > int(intervals[interval_ordinal].end) or end - start > packet_payload_bytes:
            raise ValueError("数据包越过 restart 边界或超过载荷上限")
        position = end
    if interval_ordinal != len(intervals) - 1 or position != int(intervals[-1].end):
        raise ValueError("数据包未覆盖所有 JPEG 熵字节")


def pack_packetized_frame(
    session_header: bytes | None, frame_header: bytes, jpeg_payload: bytes,
    intervals: Sequence[object], packets: Sequence[object], *,
    packet_payload_bytes: int = 1200, packet_header_bytes: int = 8,
) -> PacketizedFrame:
    """把 JPEG 结构放进可靠控制清单，把熵片段变成带序号和 CRC 的数据包。"""
    if not isinstance(frame_header, bytes) or not isinstance(jpeg_payload, bytes):
        raise TypeError("frame_header 和 jpeg_payload 必须是 bytes")
    if session_header is not None and not isinstance(session_header, bytes):
        raise TypeError("session_header 必须是 bytes 或 None")
    if not isinstance(packet_payload_bytes, int) or packet_payload_bytes <= 0:
        raise ValueError("packet_payload_bytes 必须是正整数")
    if not isinstance(packet_header_bytes, int) or packet_header_bytes < MIN_PACKET_HEADER_BYTES:
        raise ValueError("packet_header_bytes 至少为 8")
    _validate_plan(jpeg_payload, intervals, packets, packet_payload_bytes)
    session = session_header or b""
    flags = _SESSION_PRESENT if session_header is not None else 0
    prefix = _PREFIX.pack(_MAGIC, _SCHEMA, flags, len(session), len(frame_header),
                          len(jpeg_payload), len(intervals), len(packets),
                          packet_payload_bytes, packet_header_bytes)
    table = b"".join(_INTERVAL.pack(
        int(interval.start), int(interval.end), int(interval.scan), int(interval.index),
        int(interval.mcu_start), int(interval.mcu_end)) for interval in intervals)
    # gap 是 SOI/SOF/SOS/DRI/RST/EOI 等非熵字节，传输时由控制清单可靠送达。
    gaps: list[bytes] = []
    position = 0
    for interval in intervals:
        gaps.append(jpeg_payload[position:int(interval.start)])
        position = int(interval.end)
    gaps.append(jpeg_payload[position:])
    content = prefix + session + frame_header + table + b"".join(gaps)
    control = content + hashlib.sha256(content).digest()
    data_packets = []
    for index, packet in enumerate(packets):
        chunk = jpeg_payload[int(packet.start):int(packet.end)]
        # CRC 同时覆盖序号与载荷，防止序号损坏后把有效数据放进错误位置。
        index_bytes = struct.pack(">I", index)
        header = _PACKET.pack(index, zlib.crc32(index_bytes + chunk))
        data_packets.append(header + bytes(packet_header_bytes - _PACKET.size) + chunk)
    result = PacketizedFrame(control, tuple(data_packets))
    expected = (CONTROL_FIXED_BYTES + len(session) + len(frame_header) + len(jpeg_payload)
                + INTERVAL_MANIFEST_BYTES * len(intervals)
                + packet_header_bytes * len(packets))
    if result.sent_bytes != expected:
        raise AssertionError("分包字节计费与实际线格式不一致")
    return result


def receive_packetized_frame(control: bytes, received_packets: Sequence[bytes]) -> ReceivedFrame:
    """先验可靠控制，再按序号和 CRC 恢复完整区间；缺片区间交给 JPEG 重同步。"""
    if len(control) < CONTROL_FIXED_BYTES:
        raise ValueError("控制清单截断")
    if not hmac.compare_digest(hashlib.sha256(control[:-_DIGEST_BYTES]).digest(),
                               control[-_DIGEST_BYTES:]):
        raise ValueError("控制清单 SHA-256 校验失败")
    (magic, schema, flags, session_len, frame_len, jpeg_len, interval_count,
     packet_count, payload_limit, header_bytes) = _PREFIX.unpack_from(control)
    if magic != _MAGIC or schema != _SCHEMA or flags & ~_SESSION_PRESENT:
        raise ValueError("控制清单版本或标志错误")
    if not flags & _SESSION_PRESENT and session_len:
        raise ValueError("控制清单 session 长度无效")
    if not interval_count or not packet_count or not payload_limit or header_bytes < MIN_PACKET_HEADER_BYTES:
        raise ValueError("控制清单分包参数无效")
    table_end = _PREFIX.size + session_len + frame_len + interval_count * _INTERVAL.size
    if table_end > len(control) - _DIGEST_BYTES:
        raise ValueError("控制清单区间表截断")
    session_start = _PREFIX.size
    frame_start = session_start + session_len
    table_start = frame_start + frame_len
    session = control[session_start:frame_start] if flags & _SESSION_PRESENT else None
    frame = control[frame_start:table_start]
    intervals = [_INTERVAL.unpack_from(control, table_start + i * _INTERVAL.size)
                 for i in range(interval_count)]
    expected_packets: list[tuple[int, int]] = []
    previous_end = 0
    entropy_length = 0
    for ordinal, (start, end, scan, index, mcu_start, mcu_end) in enumerate(intervals):
        if start < previous_end or end <= start or end > jpeg_len or mcu_end <= mcu_start:
            raise ValueError("控制清单 JPEG 区间无效")
        length = end - start
        for offset in range(0, length, payload_limit):
            expected_packets.append((ordinal, min(payload_limit, length - offset)))
        entropy_length += length
        previous_end = end
    if len(expected_packets) != packet_count:
        raise ValueError("控制清单数据包数量无效")
    structure = control[table_end:-_DIGEST_BYTES]
    if len(structure) != jpeg_len - entropy_length:
        raise ValueError("控制清单 JPEG 结构长度无效")

    available: dict[int, bytes] = {}
    for wire_packet in received_packets:
        if len(wire_packet) < header_bytes:
            continue
        packet_index, checksum = _PACKET.unpack_from(wire_packet)
        if packet_index >= packet_count or packet_index in available:
            continue
        chunk = wire_packet[header_bytes:]
        if (len(chunk) != expected_packets[packet_index][1] or
                zlib.crc32(wire_packet[:4] + chunk) != checksum):
            continue
        available[packet_index] = chunk
    lost = tuple(index for index in range(packet_count) if index not in available)
    packets_by_interval: list[list[int]] = [[] for _ in intervals]
    for index, (ordinal, _) in enumerate(expected_packets):
        packets_by_interval[ordinal].append(index)

    # 某区间只收到部分分片时，JPEG 熵解码状态无法从中间恢复；保留下一 RST 标记。
    output: list[bytes] = []
    gap_position = 0
    previous_end = 0
    discarded = 0
    damaged_intervals = []
    for ordinal, (start, end, scan, index, _, _) in enumerate(intervals):
        gap_length = start - previous_end
        output.append(structure[gap_position:gap_position + gap_length])
        gap_position += gap_length
        ids = packets_by_interval[ordinal]
        if all(packet_index in available for packet_index in ids):
            output.extend(available[packet_index] for packet_index in ids)
        else:
            damaged_intervals.append((scan, index))
            discarded += sum(len(available[packet_index]) for packet_index in ids
                             if packet_index in available)
        previous_end = end
    output.append(structure[gap_position:])
    lost_entropy = sum(expected_packets[index][1] for index in lost)
    received_bytes = len(control) + sum(len(packet) for packet in received_packets)
    return ReceivedFrame(session, frame, b"".join(output), lost,
                         tuple(damaged_intervals), lost_entropy, discarded,
                         received_bytes)
