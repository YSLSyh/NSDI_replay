"""把 JPEG 重启区间装入应用层分组，并模拟分组丢失。

JPEG 控制标记由上层可靠传送；熵数据按重启区间分片。缺少任一分片时，
接收端丢弃该区间尚存的熵数据，在下一个 RST 标记重新同步。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image

from replay.transmission.channel import _PROFILES


@dataclass(frozen=True)
class EntropyInterval:
    """一个 JPEG 扫描中由两个重启边界围住的熵区间。"""

    scan: int
    index: int
    start: int
    end: int
    mcu_start: int
    mcu_end: int
    mcu_side: int = 8


@dataclass(frozen=True)
class EntropyPacket:
    """一个实际发送的熵数据分片；同一区间可有多个分片。"""

    scan: int
    interval: int
    interval_ordinal: int
    start: int
    end: int


@dataclass(frozen=True)
class ByteStreamOutcome:
    decoded: Image.Image | None
    received_payload: bytes
    status: str
    decode_error: str | None
    missing_pixels: np.ndarray
    lost_rows: tuple[int, ...]
    total_packets: int
    lost_packets: int
    lost_entropy_bytes: int
    packet_loss_rate: float
    byte_loss_rate: float
    target_loss_rate: float
    sent_bytes: int
    received_bytes: int
    throughput_mbps: float
    fixed_overhead_ms: float
    transfer_latency_ms: float
    proxy_kind: str = "jpeg_restart_packet_loss_v2"
    lost_packet_indices: tuple[int, ...] = ()
    lost_intervals: tuple[tuple[int, int], ...] = ()
    receiver_discarded_bytes: int = 0
    loss_model: str = "gilbert_elliott"
    restart_interval_mcus: int = 0


def _plan_jpeg(payload: bytes, packet_payload_bytes: int, *, allow_420: bool = False
               ) -> tuple[int, int, int, tuple[EntropyInterval, ...], tuple[EntropyPacket, ...]]:
    if not isinstance(packet_payload_bytes, int) or isinstance(packet_payload_bytes, bool) or packet_payload_bytes <= 0:
        raise ValueError("packet_payload_bytes 必须是正整数")
    if not isinstance(payload, bytes) or payload[:2] != b"\xff\xd8":
        raise ValueError("需要完整的 JPEG SOI 字节流")
    cursor = 2
    width = height = restart_interval = 0
    mcu_side = 8
    intervals: list[EntropyInterval] = []
    packets: list[EntropyPacket] = []
    scans = 0
    saw_eoi = False
    while cursor < len(payload):
        if payload[cursor] != 0xFF:
            raise ValueError(f"JPEG 标记边界错误，位置 {cursor}")
        while cursor < len(payload) and payload[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(payload):
            raise ValueError("JPEG 末尾缺少标记码")
        marker = payload[cursor]
        cursor += 1
        if marker == 0xD9:
            saw_eoi = True
            break
        if marker == 0x01:
            continue
        if marker in range(0xD0, 0xD9) or marker == 0x00:
            raise ValueError("JPEG 控制区出现非法 RST/SOI/转义标记")
        if cursor + 2 > len(payload):
            raise ValueError("JPEG segment 长度缺失")
        length = int.from_bytes(payload[cursor:cursor + 2], "big")
        if length < 2 or cursor + length > len(payload):
            raise ValueError("JPEG segment 长度非法")
        data = payload[cursor + 2:cursor + length]
        cursor += length
        if marker in (0xC0, 0xC1, 0xC2):
            if len(data) < 6 or len(data) != 6 + 3 * data[5]:
                raise ValueError("JPEG SOF 结构错误")
            height = int.from_bytes(data[1:3], "big")
            width = int.from_bytes(data[3:5], "big")
            if not width or not height or data[5] != 3:
                raise ValueError("仅支持非空 RGB/YCbCr 三通道 JPEG")
            sampling = tuple(data[7 + 3 * channel] for channel in range(3))
            if sampling == (0x11, 0x11, 0x11):
                mcu_side = 8
            elif allow_420 and sampling == (0x22, 0x11, 0x11):
                mcu_side = 16
            else:
                raise ValueError("分组解析要求 JPEG 4:4:4；显式 allow_420 时也接受 4:2:0")
        elif marker == 0xDD:
            if len(data) != 2:
                raise ValueError("JPEG DRI 长度错误")
            restart_interval = int.from_bytes(data, "big")
        elif marker == 0xDA:
            if not width or not height or not restart_interval:
                raise ValueError("JPEG 必须包含非零 DRI 重启间隔与受支持的 SOF")
            mcus_per_row = math.ceil(width / mcu_side)
            total_mcus = mcus_per_row * math.ceil(height / mcu_side)
            expected_intervals = math.ceil(total_mcus / restart_interval)
            scan_ranges: list[tuple[int, int]] = []
            scan_start = cursor
            next_rst = 0
            while cursor < len(payload):
                if payload[cursor] != 0xFF:
                    cursor += 1
                    continue
                marker_start = cursor
                while cursor < len(payload) and payload[cursor] == 0xFF:
                    cursor += 1
                if cursor >= len(payload):
                    raise ValueError("JPEG scan 在标记中截断")
                code = payload[cursor]
                cursor += 1
                if code == 0x00:  # FF00 是熵字节转义，不能作为分组边界。
                    continue
                if 0xD0 <= code <= 0xD7:
                    if code != 0xD0 + next_rst % 8:
                        raise ValueError("JPEG RST 序号不连续")
                    scan_ranges.append((scan_start, marker_start))
                    scan_start = cursor
                    next_rst += 1
                    continue
                scan_ranges.append((scan_start, marker_start))
                cursor = marker_start
                break
            else:
                raise ValueError("JPEG scan 缺少结尾标记")
            if len(scan_ranges) != expected_intervals:
                raise ValueError("JPEG scan 的重启区间数与 DRI/图像尺寸不符")
            for index, (start, end) in enumerate(scan_ranges):
                if end < start:
                    raise ValueError("JPEG scan 熵区间边界错误")
                interval = EntropyInterval(
                    scans, index, start, end,
                    index * restart_interval,
                    min((index + 1) * restart_interval, total_mcus),
                    mcu_side,
                )
                ordinal = len(intervals)
                intervals.append(interval)
                # 分片边界由编码区间与 MTU 决定；丢失事件仍独立作用于每片。
                for part_start in range(start, end, packet_payload_bytes):
                    packets.append(EntropyPacket(
                        scans, index, ordinal, part_start,
                        min(part_start + packet_payload_bytes, end),
                    ))
            scans += 1
    if not saw_eoi or cursor != len(payload) or not packets:
        raise ValueError("JPEG 缺少 EOI 或可分组熵数据")
    return width, height, restart_interval, tuple(intervals), tuple(packets)


def plan_entropy_packets(
    payload: bytes, packet_payload_bytes: int = 1200, *, allow_420: bool = False,
) -> tuple[int, int, tuple[EntropyInterval, ...], tuple[EntropyPacket, ...]]:
    """返回图像尺寸、重启区间及按区间顺序发送的实际数据分片。"""
    width, height, _, intervals, packets = _plan_jpeg(
        payload, packet_payload_bytes, allow_420=allow_420)
    return width, height, intervals, packets


def count_entropy_packets(payload: bytes, packet_payload_bytes: int = 1200,
                          *, allow_420: bool = False) -> int:
    """返回实际熵数据分片数，供编码预算计入每包开销。"""
    return len(plan_entropy_packets(payload, packet_payload_bytes,
                                    allow_420=allow_420)[3])


def count_entropy_intervals(payload: bytes, packet_payload_bytes: int = 1200,
                            *, allow_420: bool = False) -> int:
    """返回真实 DRI/RST 定义的区间数，供可靠 manifest 计费。"""
    return len(plan_entropy_packets(payload, packet_payload_bytes,
                                    allow_420=allow_420)[2])


def _uniform(seed: int, image_id: str, scenario: str, purpose: str, index: int = 0) -> float:
    value = json.dumps([seed, image_id, scenario, purpose, index],
                       ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(sha256(value).digest()[:8], "big") / 2**64


def _packet_loss_events(
    count: int, rate: float, seed: int, image_id: str, scenario: str,
    loss_model: str, loss_trace: Sequence[bool] | None, mean_burst_packets: float,
) -> tuple[tuple[bool, ...], str]:
    if loss_trace is not None:
        if len(loss_trace) < count or any(type(event) is not bool for event in loss_trace[:count]):
            raise ValueError("外部 trace 必须至少包含每个发送分片的布尔丢包事件")
        return tuple(loss_trace[:count]), "external_trace"
    if loss_model == "bernoulli":
        return tuple(_uniform(seed, image_id, scenario, "packet_loss", index) < rate
                     for index in range(count)), "bernoulli"
    if loss_model != "gilbert_elliott":
        raise ValueError("loss_model 仅支持 gilbert_elliott 或 bernoulli")
    if not math.isfinite(mean_burst_packets) or mean_burst_packets < 1:
        raise ValueError("mean_burst_packets 必须为不小于 1 的有限数")
    # 两状态 GE：坏状态必丢、好状态必收。稳态坏状态概率为目标丢包率。
    bad_to_good = 1.0 / mean_burst_packets
    good_to_bad = rate * bad_to_good / (1.0 - rate)
    if good_to_bad > 1:
        raise ValueError("给定丢包率和平均连续丢包长度无法构成两状态模型")
    bad = _uniform(seed, image_id, scenario, "ge_initial") < rate
    events: list[bool] = []
    for index in range(count):
        if index:
            transition = _uniform(seed, image_id, scenario, "ge_transition", index)
            bad = transition >= bad_to_good if bad else transition < good_to_bad
        events.append(bad)
    return tuple(events), "gilbert_elliott"


def simulate_restart_stream_loss(
    payload: bytes, *, image_id: str, scenario: str, level: str, seed: int,
    packet_header_bytes: int = 8, additional_header_bytes: int = 0,
    fixed_overhead_ms: float = 10.0, packet_payload_bytes: int = 1200,
    loss_model: str = "gilbert_elliott", loss_trace: Sequence[bool] | None = None,
    mean_burst_packets: float = 3.0,
) -> ByteStreamOutcome:
    """按应用层分片丢包，再依 JPEG 重启区间进行接收端丢弃与解码。

    合成事件按分片索引确定，所有方法共享同一图像/场景/种子下的事件前缀；
    不把合成 GE/Bernoulli 事件称为实测网络轨迹。
    """
    if scenario not in _PROFILES or level not in _PROFILES[scenario]["loss_rates"]:
        raise ValueError(f"未知场景或丢包级别：{scenario}/{level}")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id 必须是非空字符串")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0
           for x in (packet_header_bytes, additional_header_bytes)):
        raise ValueError("传输头字节数必须是非负整数")
    if not math.isfinite(fixed_overhead_ms) or fixed_overhead_ms < 0:
        raise ValueError("固定时延必须是有限非负数")
    width, height, restart_interval, intervals, packets = _plan_jpeg(payload, packet_payload_bytes)
    rate = float(_PROFILES[scenario]["loss_rates"][level])
    events, actual_model = _packet_loss_events(
        len(packets), rate, seed, image_id, scenario,
        loss_model, loss_trace, mean_burst_packets,
    )
    lost_packet_indices = tuple(index for index, dropped in enumerate(events) if dropped)
    lost_packets = tuple(packets[index] for index in lost_packet_indices)
    lost_keys = {(packet.scan, packet.interval) for packet in lost_packets}
    lost_intervals = tuple(interval for interval in intervals
                           if (interval.scan, interval.index) in lost_keys)
    lost_interval_ids = tuple((interval.scan, interval.index) for interval in lost_intervals)
    lost_entropy_bytes = sum(packet.end - packet.start for packet in lost_packets)
    receiver_discarded_bytes = (sum(interval.end - interval.start for interval in lost_intervals)
                                - lost_entropy_bytes)
    if receiver_discarded_bytes < 0:
        raise AssertionError("接收端丢弃字节数不能为负")

    missing = np.zeros((height, width), dtype=np.float32)
    mcus_per_row = math.ceil(width / 8)
    affected_mcu_rows: set[int] = set()
    for interval in lost_intervals:
        for mcu in range(interval.mcu_start, interval.mcu_end):
            row, col = divmod(mcu, mcus_per_row)
            missing[row * 8:min((row + 1) * 8, height),
                    col * 8:min((col + 1) * 8, width)] = 1.0
            affected_mcu_rows.add(row)

    # 只删除损坏区间的熵字节；可靠传送的结构标记与 RST 保持原位置和顺序。
    chunks: list[bytes] = []
    position = 0
    for interval in lost_intervals:
        chunks.append(payload[position:interval.start])
        position = interval.end
    chunks.append(payload[position:])
    received_payload = b"".join(chunks)
    decoded: Image.Image | None = None
    decode_error: str | None = None
    try:
        with Image.open(BytesIO(received_payload)) as image:
            image.load()
            decoded = image.convert("RGB")
            if decoded.size != (width, height):
                raise ValueError("接收图像尺寸错误")
    except Exception as exc:
        decoded = None
        decode_error = f"{type(exc).__name__}: {exc}"

    low, high = _PROFILES[scenario]["throughput_mbps"]
    throughput = low + (high - low) * _uniform(seed, image_id, scenario, "throughput")
    sent_bytes = len(payload) + len(packets) * packet_header_bytes + additional_header_bytes
    received_bytes = sent_bytes - lost_entropy_bytes - len(lost_packets) * packet_header_bytes
    latency = fixed_overhead_ms + sent_bytes * 8 / (throughput * 1_000_000) * 1_000
    return ByteStreamOutcome(
        decoded=decoded, received_payload=received_payload,
        status="ok" if decoded is not None else "undecodable",
        decode_error=decode_error, missing_pixels=missing,
        lost_rows=tuple(sorted(affected_mcu_rows)),
        total_packets=len(packets), lost_packets=len(lost_packets),
        lost_entropy_bytes=lost_entropy_bytes,
        packet_loss_rate=len(lost_packets) / len(packets),
        byte_loss_rate=lost_entropy_bytes / len(payload), target_loss_rate=rate,
        sent_bytes=sent_bytes, received_bytes=received_bytes,
        throughput_mbps=throughput, fixed_overhead_ms=fixed_overhead_ms,
        transfer_latency_ms=latency,
        lost_packet_indices=lost_packet_indices,
        lost_intervals=lost_interval_ids,
        receiver_discarded_bytes=receiver_discarded_bytes,
        loss_model=actual_model, restart_interval_mcus=restart_interval,
    )
