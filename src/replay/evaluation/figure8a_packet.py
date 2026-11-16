"""Cityscapes 图 8(a) 的原尺寸、逐包轨迹、等 wire 字节对照实验。

这是论文约束下的项目实现。JPEG 重启分包、可靠控制清单和 LimitNet
分层封包均由 replay 定义；论文没有公开这些线格式或原始 5G 轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
import struct
from typing import Callable
import zlib

import numpy as np
import torch
from PIL import Image

from replay.baselines.limitnet import LimitNetCodec, LimitNetStream, load_limitnet
from replay.core.assets import verify_model_lock
from replay.core.config import ReplayConfig, write_json
from replay.data.cityscapes import cityscapes_pairs, load_cityscapes_pair
from replay.evaluation.figure8 import binary_counts, ground_truth_patches
from replay.evaluation.figure8a_sender_report import load_dct_sender_candidate
from replay.transmission.baselines import encode_baseline_under_budget
from replay.transmission.header import FrameHeader, SessionHeader
from replay.transmission.jpeg import encode_under_budget
from replay.transmission.packet_loss import plan_entropy_packets
from replay.transmission.packet_trace import (PacketTrace, load_packet_trace_csv,
                                              load_trace_provenance_json)
from replay.transmission.packet_wire import (CONTROL_FIXED_BYTES, INTERVAL_MANIFEST_BYTES,
                                             PacketizedFrame, pack_packetized_frame,
                                             receive_packetized_frame)


TARGETS = {"car": 13, "pedestrian": 11, "vegetation": 8}
LIMITNET_MODES = ("official224", "native_adapted")


def _validate_limitnet_checkpoint_metadata(mode: str, metadata: dict) -> str:
    """将输入分辨率与训练来源绑定，防止把未微调权重当论文基线。"""
    methods_for_limitnet_mode(mode)
    source = str(metadata.get("source", "unrecorded"))
    if mode == "official224":
        if source == "official_limitnet_imagenet_pretrained":
            return "official_imagenet_pretrained"
        if (source == "cityscapes_finetuned_from_official" and
                metadata.get("training_dataset") == "Cityscapes train" and
                metadata.get("input_mode") == "official224_full_frame" and
                metadata.get("codec_input_resolution") == [224, 224] and
                metadata.get("pixel_rescaling") is True and
                type(metadata.get("epochs_completed")) is int and
                metadata["epochs_completed"] >= 1):
            return "cityscapes_train_finetuned_224"
        raise ValueError("official224 要求官方 ImageNet 权重或其 Cityscapes train "
                         "固定 224×224 整帧微调权重；"
                         f"当前 checkpoint source={source}")
    if (source.startswith("cityscapes_finetuned") and
            metadata.get("training_dataset") == "Cityscapes train" and
            metadata.get("pixel_rescaling") is False and
            metadata.get("input_mode", "native_crop") == "native_crop" and
            type(metadata.get("epochs_completed")) is int and
            metadata["epochs_completed"] >= 1):
        return "cityscapes_train_finetuned_native"
    raise ValueError("native_adapted 要求记录 Cityscapes train 原生尺度微调来源与轮数的权重")


def methods_for_limitnet_mode(mode: str) -> tuple[str, ...]:
    if mode not in LIMITNET_MODES:
        raise ValueError(f"未知 LimitNet 评测模式：{mode}")
    return ("jpeg", f"limitnet_{mode}", "kdc_direct", "kdc_restored")


METHODS = methods_for_limitnet_mode("official224")
PACKET_PAYLOAD_BYTES = 1200
PACKET_HEADER_BYTES = 8
TRACE_RESERVED_WINDOW_PACKETS = 256
_LIM_MAGIC = b"RLIMPKT1"
_LIM_MAGIC_PARITY = b"RLIMPKT2"
_LIM_PREFIX = struct.Struct(">8sIII")  # magic、潜变量头长度、层数、分片载荷上限
_LIM_LAYER = struct.Struct(">II")  # 该层字节数、分片数
_LIM_PACKET = struct.Struct(">II")  # 全局序号、CRC32


def _validate_packet_adapted_receiver_metadata(
    metadata: dict, *, task: str, budget_bytes: int, trace_sha256: str,
    sender_tuning_sha256: str | None, codec_candidate: dict | None,
    train_image_ids: set[str], tuning_image_ids: set[str],
    initial_receiver_sha256: str, source_packet_bytes: int | None,
) -> dict:
    """只接受同一 TRAIN 发送端/轨迹下的独立 4:2:0 丢包接收端消融。"""
    if not isinstance(metadata, dict):
        raise ValueError("适配接收端 checkpoint 缺少 metadata")
    if sender_tuning_sha256 is None or codec_candidate is None:
        raise ValueError("适配接收端必须与已核验的 TRAIN 多维发送端报告配对")
    if codec_candidate.get("subsampling") != 2:
        raise ValueError("适配接收端仅支持当前训练报告选中的 4:2:0 candidate")
    if (metadata.get("adaptation") !=
            "Cityscapes train measured-loss conditional packet receiver" or
            metadata.get("not_original_paper_pretraining") is not True or
            metadata.get("cityscapes_split") != "train only"):
        raise ValueError("适配接收端必须明确标记为 Cityscapes TRAIN-only 消融")
    if (type(metadata.get("budget_wire_bytes")) is not int or
            metadata["budget_wire_bytes"] != budget_bytes or budget_bytes != 30_000):
        raise ValueError("适配接收端与评测必须使用相同的 30 kB wire 预算")
    if metadata.get("trace_csv_sha256") != trace_sha256:
        raise ValueError("适配接收端的逐包轨迹 CSV 哈希与评测不一致")
    if (source_packet_bytes is not None and
            metadata.get("trace_source_packet_bytes") != source_packet_bytes):
        raise ValueError("适配接收端的轨迹源包长与评测不一致")
    if metadata.get("transmitted_packet_payload_bytes") != PACKET_PAYLOAD_BYTES:
        raise ValueError("适配接收端的应用分片载荷与评测不一致")
    selection = metadata.get("sender_selection")
    if (not isinstance(selection, dict) or
            selection.get("mode") != "train_tuned_multidimensional" or
            selection.get("sha256") != sender_tuning_sha256):
        raise ValueError("适配接收端的 TRAIN 发送端报告哈希与评测不一致")
    selected_by_task = selection.get("candidates_by_task")
    recorded_by_task = metadata.get("codec_candidate_by_task")
    if (not isinstance(selected_by_task, dict) or
            not isinstance(recorded_by_task, dict) or
            selected_by_task.get(task) != codec_candidate or
            recorded_by_task.get(task) != codec_candidate):
        raise ValueError("适配接收端的 4:2:0 codec candidate 与评测不一致")
    if metadata.get("initial_receiver_sha256") != initial_receiver_sha256:
        raise ValueError("适配接收端并非从当前配置的原始接收端权重初始化")
    trained = metadata.get("training_image_ids")
    held_out = metadata.get("holdout_image_ids")
    if (not isinstance(trained, list) or not isinstance(held_out, list) or
            not trained or not held_out or
            not all(isinstance(value, str) for value in trained + held_out) or
            len(set(trained)) != len(trained) or len(set(held_out)) != len(held_out) or
            not set(trained).isdisjoint(held_out) or
            not set(trained).issubset(train_image_ids) or
            not set(held_out).issubset(train_image_ids) or
            not set(held_out).isdisjoint(tuning_image_ids) or
            metadata.get("holdout_excludes_sender_tuning_images") is not True):
        raise ValueError("适配接收端训练/早停样本必须全部来自 Cityscapes TRAIN，且早停不能碰发送端调参图")
    epoch = metadata.get("epoch_completed")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("适配接收端 checkpoint 未记录已完成的训练轮次")
    return {
        "training_image_count": len(trained),
        "holdout_train_image_count": len(held_out),
        "epoch_completed": epoch,
        "sender_tuning_report_sha256": sender_tuning_sha256,
        "trace_csv_sha256": trace_sha256,
        "codec_candidate": codec_candidate,
        "initial_receiver_sha256": initial_receiver_sha256,
    }


def _limitnet_input(image: Image.Image, mode: str) -> Image.Image:
    """官方模式沿用固定 224×224 入口；原生适配模式不缩放。"""
    methods_for_limitnet_mode(mode)
    return (image.resize((224, 224), Image.Resampling.BILINEAR)
            if mode == "official224" else image)


def _limitnet_output(decoded: Image.Image, original_size: tuple[int, int],
                     mode: str) -> Image.Image:
    """只把固定尺寸基线的解码图放大到统一分割头的 Cityscapes 画布。"""
    expected = (224, 224) if mode == "official224" else original_size
    if decoded.size != expected:
        raise ValueError("LimitNet 解码尺寸与所选模式不一致")
    return (decoded.resize(original_size, Image.Resampling.BILINEAR)
            if mode == "official224" else decoded)


@dataclass(frozen=True)
class _LimitWire:
    """独立层可重组的真实字节封包；控制清单假定可靠到达。"""

    control: bytes
    packets: tuple[bytes, ...]

    @property
    def sent_bytes(self) -> int:
        return len(self.control) + sum(map(len, self.packets))


def _limit_wire(stream: LimitNetStream, count: int, *,
                parity_packets_per_layer: int = 0) -> _LimitWire:
    if not 0 <= count <= len(stream.layers):
        raise ValueError("LimitNet 层数越界")
    if parity_packets_per_layer not in (0, 1):
        raise ValueError("LimitNet 每层校验包数只支持 0 或 1")
    manifest = []
    packets = []
    for layer in stream.layers[:count]:
        parts = [layer[start:start + PACKET_PAYLOAD_BYTES]
                 for start in range(0, len(layer), PACKET_PAYLOAD_BYTES)]
        if not parts:
            raise ValueError("LimitNet 不能发送空潜变量层")
        manifest.append(_LIM_LAYER.pack(len(layer), len(parts)))
        for part in parts:
            index = len(packets)
            index_bytes = struct.pack(">I", index)
            packets.append(_LIM_PACKET.pack(index, zlib.crc32(index_bytes + part)) + part)
        if parity_packets_per_layer:
            # 单个异或校验包可恢复本层任意一个丢失的数据包；末包按原长裁回。
            parity = bytearray(max(map(len, parts)))
            for part in parts:
                for offset, value in enumerate(part):
                    parity[offset] ^= value
            index = len(packets)
            index_bytes = struct.pack(">I", index)
            body = bytes(parity)
            packets.append(_LIM_PACKET.pack(index, zlib.crc32(index_bytes + body)) + body)
    magic = _LIM_MAGIC_PARITY if parity_packets_per_layer else _LIM_MAGIC
    content = (_LIM_PREFIX.pack(magic, len(stream.header), count,
                                PACKET_PAYLOAD_BYTES) + stream.header + b"".join(manifest))
    return _LimitWire(content + sha256(content).digest(), tuple(packets))


def _limit_under_budget(codec: LimitNetCodec, image: Image.Image,
                        focus: np.ndarray | None, budget: int, *,
                        parity_packets_per_layer: int = 0) -> tuple[_LimitWire, int]:
    stream = codec.encode_full(image, focus_pixels=focus)
    best = _limit_wire(stream, 0, parity_packets_per_layer=parity_packets_per_layer)
    if best.sent_bytes > budget:
        raise ValueError(f"LimitNet 头信息需 {best.sent_bytes} 字节，超过预算 {budget}")
    count = 0
    for candidate in range(1, len(stream.layers) + 1):
        wire = _limit_wire(stream, candidate,
                           parity_packets_per_layer=parity_packets_per_layer)
        if wire.sent_bytes > budget:
            break
        best, count = wire, candidate
    if count == 0:
        raise ValueError(f"LimitNet 在 {budget} 字节预算内无法送达任何完整潜变量层")
    return best, count


def _receive_limit(wire: _LimitWire, lost: tuple[bool, ...],
                   codec: LimitNetCodec) -> tuple[Image.Image, int, int]:
    """CRC/序号验包；损坏层舍弃，其余完整层按独立掩码重组。"""
    if len(lost) != len(wire.packets):
        raise ValueError("LimitNet 丢包轨迹长度错误")
    control = wire.control
    if len(control) < _LIM_PREFIX.size + 32 or sha256(control[:-32]).digest() != control[-32:]:
        raise ValueError("LimitNet 控制清单校验失败")
    magic, header_length, layer_count, payload_limit = _LIM_PREFIX.unpack_from(control)
    if magic not in (_LIM_MAGIC, _LIM_MAGIC_PARITY) or payload_limit != PACKET_PAYLOAD_BYTES:
        raise ValueError("LimitNet 分包版本错误")
    parity_packets_per_layer = int(magic == _LIM_MAGIC_PARITY)
    manifest_start = _LIM_PREFIX.size + header_length
    if manifest_start + layer_count * _LIM_LAYER.size + 32 != len(control):
        raise ValueError("LimitNet 控制清单长度错误")
    model_header = control[_LIM_PREFIX.size:manifest_start]
    available: dict[int, bytes] = {}
    for index, (packet, dropped) in enumerate(zip(wire.packets, lost)):
        if dropped or len(packet) < _LIM_PACKET.size:
            continue
        number, checksum = _LIM_PACKET.unpack_from(packet)
        body = packet[_LIM_PACKET.size:]
        if number == index and zlib.crc32(packet[:4] + body) == checksum:
            available[index] = body
    accepted = []
    cursor = 0
    for layer_index in range(layer_count):
        layer_size, n_parts = _LIM_LAYER.unpack_from(control, manifest_start +
                                                     layer_index * _LIM_LAYER.size)
        if layer_size <= 0 or n_parts != (layer_size + payload_limit - 1) // payload_limit:
            raise ValueError("LimitNet 层大小与数据分片数不一致")
        parts = [available.get(index) for index in range(cursor, cursor + n_parts)]
        for part_index, part in enumerate(parts):
            expected = min(payload_limit, layer_size - part_index * payload_limit)
            if part is not None and len(part) != expected:
                raise ValueError("LimitNet 数据分片长度错误")
        if parity_packets_per_layer:
            parity = available.get(cursor + n_parts)
            if parity is not None and len(parity) != min(payload_limit, layer_size):
                raise ValueError("LimitNet 校验分片长度错误")
            missing = [index for index, part in enumerate(parts) if part is None]
            if len(missing) == 1 and parity is not None:
                recovered = bytearray(parity)
                for part in parts:
                    if part is not None:
                        for offset, value in enumerate(part):
                            recovered[offset] ^= value
                part_index = missing[0]
                expected = min(payload_limit, layer_size - part_index * payload_limit)
                parts[part_index] = bytes(recovered[:expected])
        if all(part is not None for part in parts):
            layer = b"".join(parts)
            if len(layer) != layer_size:
                raise ValueError("LimitNet 分层字节长度错误")
            accepted.append(layer)
        # 每层自带位置掩码，后续完整层不依赖前一层；丢包只影响所在层。
        cursor += n_parts + parity_packets_per_layer
    if cursor != len(wire.packets):
        raise ValueError("LimitNet 清单与实际分片数不一致")
    decoded = codec.decode_bytes(model_header + b"".join(accepted))
    received_bytes = len(control) + sum(len(packet) for packet, dropped
                                        in zip(wire.packets, lost) if not dropped)
    return decoded, len(accepted), received_bytes


def _jpeg_wire(image: Image.Image, *, focus: np.ndarray | None,
               session: bytes | None, frame: bytes, budget: int,
               config: ReplayConfig,
               background_lowpass_sigma: float = 0.0,
               codec_candidate: dict | None = None) -> tuple[PacketizedFrame, bytes, dict]:
    image_cfg = config.section("image")
    importance = config.section("importance")
    # RST2048 来自 20 张 Cityscapes train 的可行性筛选，论文未公开此参数。
    restart_blocks = (2048 if codec_candidate is None else
                      int(codec_candidate["restart_blocks"]))
    subsampling = (0 if codec_candidate is None else
                   int(codec_candidate["subsampling"]))
    allow_420 = subsampling == 2
    remaining = budget - CONTROL_FIXED_BYTES - len(session or b"") - len(frame)
    if remaining <= 0:
        raise ValueError("控制头已耗尽 30 kB 预算")
    common = dict(restart_marker_blocks=restart_blocks,
                  packet_header_bytes=PACKET_HEADER_BYTES,
                  packet_manifest_bytes_per_interval=INTERVAL_MANIFEST_BYTES,
                  optimize_huffman=True, allow_420=allow_420)
    if focus is None:
        payload, info = encode_baseline_under_budget(
            image, "jpeg", remaining, subsampling, **common)
    else:
        payload, info = encode_under_budget(
            image, focus, int(image_cfg["patch_size"]), remaining, subsampling,
            (int(importance["dct_high_frequency_cutoff"]) if codec_candidate is None
             else int(codec_candidate["cutoff"])),
            luma_strength=(float(importance["prequant_luma_strength"])
                           if codec_candidate is None else float(codec_candidate["strength"])),
            chroma_strength=(float(importance["prequant_chroma_strength"])
                             if codec_candidate is None else float(codec_candidate["strength"])),
            gamma=float(importance["prequant_gamma"]),
            importance_ratio_low=float(importance["importance_ratio_low"]),
            importance_ratio_high=float(importance["importance_ratio_high"]),
            background_lowpass_sigma=(background_lowpass_sigma if codec_candidate is None
                                      else float(codec_candidate["sigma"])),
            roi_halo_pixels=(0.0 if codec_candidate is None else
                             float(codec_candidate["roi_halo_pixels"])),
            zero_threshold_multiplier=(0.0 if codec_candidate is None else
                                       float(codec_candidate["zero_threshold_multiplier"])),
            fast_prequant=True, **common)
    width, height, intervals, packets = plan_entropy_packets(
        payload, PACKET_PAYLOAD_BYTES, allow_420=allow_420)
    if (width, height) != image.size:
        raise AssertionError("JPEG 编码时改变了 Cityscapes 原图分辨率")
    wire = pack_packetized_frame(session, frame, payload, intervals, packets,
                                 packet_payload_bytes=PACKET_PAYLOAD_BYTES,
                                 packet_header_bytes=PACKET_HEADER_BYTES)
    if wire.sent_bytes > budget or wire.sent_bytes != (CONTROL_FIXED_BYTES +
            len(session or b"") + len(frame) + info["codec_wire_bytes"]):
        raise AssertionError("JPEG 实际 wire 字节数与预算搜索不一致")
    return wire, payload, info


def _decode_jpeg_loss(wire: PacketizedFrame, payload: bytes, loss: tuple[bool, ...],
                      *, budget: int, allow_420: bool = False
                      ) -> tuple[Image.Image | None, np.ndarray, dict]:
    """仅对真实分包码流丢失字节，按接收端 RST 边界重同步解码。"""
    if len(loss) != len(wire.packets):
        raise ValueError("JPEG 丢包轨迹长度错误")
    received = receive_packetized_frame(
        wire.control, [packet for packet, dropped in zip(wire.packets, loss) if not dropped])
    expected_lost = tuple(index for index, dropped in enumerate(loss) if dropped)
    if received.lost_packet_indices != expected_lost or wire.sent_bytes > budget:
        raise AssertionError("逐包接收结果与发送分片或总预算不一致")
    width, height, intervals, _ = plan_entropy_packets(
        payload, PACKET_PAYLOAD_BYTES, allow_420=allow_420)
    missing = np.zeros((height, width), dtype=np.float32)
    lost_intervals = set(received.lost_intervals)
    mcu_side = intervals[0].mcu_side
    if any(item.mcu_side != mcu_side for item in intervals):
        raise ValueError("JPEG 扫描中的 MCU 尺寸不一致")
    mcus_per_row = math.ceil(width / mcu_side)
    for interval in intervals:
        if (interval.scan, interval.index) not in lost_intervals:
            continue
        for mcu in range(interval.mcu_start, interval.mcu_end):
            row, col = divmod(mcu, mcus_per_row)
            missing[row * mcu_side:min((row + 1) * mcu_side, height),
                    col * mcu_side:min((col + 1) * mcu_side, width)] = 1.0
    candidate = None
    decode_error = None
    try:
        with Image.open(BytesIO(received.jpeg_payload)) as decoded:
            decoded.load()
            candidate = decoded.convert("RGB")
        if candidate.size != (width, height):
            raise ValueError("JPEG 解码尺寸与原始码流不一致")
    except Exception as error:
        candidate = None
        decode_error = f"{type(error).__name__}: {error}"
    fields = {"sent_bytes": wire.sent_bytes, "received_bytes": received.received_bytes,
              "jpeg_bytes": len(payload), "packet_count": len(wire.packets),
              "lost_packets": len(expected_lost),
              "lost_entropy_bytes": received.lost_entropy_bytes,
              "lost_intervals": len(received.lost_intervals),
              "receiver_discarded_bytes": received.receiver_discarded_bytes,
              "loss_rate": len(expected_lost) / len(wire.packets),
              "decode_error": decode_error,
              "channel_kind": "measured_trace_on_restart_jpeg_entropy_packets"}
    return candidate, missing, fields


def _digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _wire_sha256(wire: _LimitWire | PacketizedFrame) -> str:
    """保存实际发送控制字节和逐包内容的指纹，供严格重放定位偏差。"""
    result = sha256()
    result.update(struct.pack(">I", len(wire.control)))
    result.update(wire.control)
    for packet in wire.packets:
        result.update(struct.pack(">I", len(packet)))
        result.update(packet)
    return result.hexdigest()


def _image_sha256(image: Image.Image) -> str:
    """量化后接收图的 RGB 像素指纹，独立于 PNG/JPEG 文件写法。"""
    rgb = image.convert("RGB")
    result = sha256(struct.pack(">II", *rgb.size))
    result.update(rgb.tobytes())
    return result.hexdigest()


def _load_sender_tuning_sigma(config: ReplayConfig, report_path: Path, *,
                              task: str, budget_bytes: int,
                              train_image_ids: set[str]) -> tuple[float, str]:
    """只读取训练集调参结果；完整性、codec 条件和来源不符则拒绝正式评测。"""
    raw = report_path.read_bytes()
    digest = sha256(raw).hexdigest()
    report = json.loads(raw.decode("utf-8"))
    settings = report.get("settings", {})
    required = {
        "protocol": "figure8a_sender_train_tuning_v1", "split": "train",
        "budget_wire_bytes": 30_000, "native_resolution": [2048, 1024],
        "jpeg_subsampling": "4:4:4", "optimize_huffman": True,
        "restart_blocks": 2048, "fast_prequant": True,
        "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
        "packet_header_bytes": PACKET_HEADER_BYTES,
        "segmenter_revision": config.section("models")["segmenter"]["revision"],
        "config_sha256": sha256(json.dumps(config.values, sort_keys=True,
                                            default=str).encode("utf-8")).hexdigest(),
    }
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("训练集调参报告的划分、预算、模型或 codec 配置与正式评测不一致")
    if budget_bytes != 30_000 or settings.get("pilot") is not False:
        raise ValueError("正式图8(a)只接受 30,000 字节预算和非 pilot 的训练集报告")
    sample_count = settings.get("sample_images_per_task")
    if type(sample_count) is not int or sample_count < 50:
        raise ValueError("训练集调参报告至少需要每类 50 张图像")
    tasks = settings.get("tasks")
    sampled = settings.get("sampled_image_ids")
    sigmas = settings.get("sigmas")
    if (not isinstance(tasks, list) or set(tasks) != set(TARGETS) or
            not isinstance(sampled, dict) or set(sampled) != set(TARGETS) or
            not isinstance(sigmas, list) or not sigmas):
        raise ValueError("训练集调参报告缺少三个类别或 sigma 候选")
    sigma_values = [float(value) for value in sigmas]
    if (len(set(sigma_values)) != len(sigma_values) or
            any(not math.isfinite(value) or value <= 0 for value in sigma_values)):
        raise ValueError("训练集调参报告的 sigma 候选无效")
    for name in TARGETS:
        image_ids = sampled[name]
        if (not isinstance(image_ids, list) or len(image_ids) != sample_count or
                len(set(image_ids)) != sample_count or
                not set(image_ids).issubset(train_image_ids)):
            raise ValueError(f"{name} 的 50+ 张调参图像未能在 Cityscapes train 核实")
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("训练集调参报告缺少逐图记录")
    seen: dict[tuple[str, float], set[str]] = {
        (name, sigma): set() for name in TARGETS for sigma in sigma_values}
    for row in runs:
        if not isinstance(row, dict) or row.get("task") not in TARGETS:
            raise ValueError("训练集调参报告含未知任务或非法逐图记录")
        key = row["task"], float(row.get("sigma"))
        image_id = row.get("image_id")
        if (key not in seen or image_id not in sampled[key[0]] or
                image_id in seen[key] or row.get("status") not in {"ok", "infeasible"}):
            raise ValueError("训练集调参报告的逐图记录缺失、重复或来自错误样本池")
        seen[key].add(image_id)
    if any(seen[(name, sigma)] != set(sampled[name])
           for name in TARGETS for sigma in sigma_values):
        raise ValueError("训练集调参报告没有完整覆盖每类每个 sigma 的训练样本")
    summary = report.get("summary", {})
    if not isinstance(summary, dict) or set(summary) != set(TARGETS):
        raise ValueError("训练集调参报告缺少三个类别的选择结果")
    for name in TARGETS:
        selection = summary[name]
        if not isinstance(selection, dict) or not isinstance(selection.get("candidates"), dict):
            raise ValueError("训练集调参报告的候选统计不完整")
        chosen = float(selection.get("selected_sigma"))
        if chosen not in sigma_values or str(chosen) not in selection["candidates"]:
            raise ValueError("训练集调参报告选择了未评测的 sigma")
    return float(summary[task]["selected_sigma"]), digest


def _fixed_trace_start(trace: PacketTrace, *, seed: int, image_id: str) -> int:
    """同一张图的三个任务和全部 codec 都使用同一轨迹起点。"""
    if len(trace.events) < TRACE_RESERVED_WINDOW_PACKETS:
        raise ValueError(f"5G 轨迹少于固定保留窗口 {TRACE_RESERVED_WINDOW_PACKETS} 包")
    key = json.dumps([seed, image_id, trace.sha256],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    choices = len(trace.events) - TRACE_RESERVED_WINDOW_PACKETS + 1
    return int.from_bytes(sha256(key).digest()[:8], "big") % choices


def _trace_packet_size_policy(capture_method: str, *, formal: bool,
                              allow_mismatch: bool) -> tuple[int | None, bool | None, str]:
    """源包长与本项目 1200B 应用载荷不同，只能解释为丢包模式压力测试。"""
    found = re.findall(r"(?<![a-z0-9_])source_packet_bytes\s*=\s*(\d+)\b",
                       capture_method.casefold())
    if len(set(found)) > 1:
        raise ValueError("轨迹来源对 source_packet_bytes 给出了冲突的数值")
    source_bytes = int(found[0]) if found else None
    if source_bytes is not None and source_bytes <= 0:
        raise ValueError("source_packet_bytes 必须为正")
    mismatch = None if source_bytes is None else source_bytes != PACKET_PAYLOAD_BYTES
    if formal and mismatch is not False and not allow_mismatch:
        raise ValueError("正式 500 张评测的轨迹源包长未知或不同于 1200B；"
                         "如需丢包模式压力测试，请显式传 --allow-packet-size-mismatch")
    interpretation = (
        "declared source packet size equals the 1200B application payload; "
        "this does not establish identical network or vehicular conditions"
        if mismatch is False else
        "measured packet-loss pattern stress test on 1200B application fragments; "
        "not a source-MTU-matched or original vehicular trace replay")
    return source_bytes, mismatch, interpretation


def _is_budget_error(error: ValueError) -> bool:
    return "预算" in str(error) or "无法满足" in str(error)


def _kdc_restored_counts(direct_counts: list[int], lost_intervals: int,
                         missing_pixels: np.ndarray,
                         restore_counts: Callable[[], list[int]]) -> tuple[list[int], bool]:
    """没有真实缺失时，原样使用直解分割计数且完全不调用恢复网络。"""
    if lost_intervals == 0 or not np.any(missing_pixels):
        return direct_counts.copy(), False
    return restore_counts(), True


def _summarize(rows: list[dict], task: str, seed: int,
               expected_image_ids: list[str],
               methods: tuple[str, ...] = METHODS) -> dict:
    """完整验证集主指标：失败帧目标像素全算 FN，绝不从分母删掉。"""
    by_id: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_id.setdefault(row["image_id"], {})[row["method"]] = row
    ready = [image_id for image_id in expected_image_ids
             if all(method in by_id.get(image_id, {}) for method in methods)]
    complete = len(ready) == len(expected_image_ids)
    paired = [by_id[image_id] for image_id in ready
              if all(by_id[image_id][method]["status"] == "ok" for method in methods)]
    generator = np.random.default_rng(seed)
    # 同一批图像索引和同一组 bootstrap 抽样用于四方法，保持配对比较。
    resampling = (generator.integers(0, len(ready), size=(1000, len(ready)))
                  if complete and ready else None)
    methods_out = {}
    for method in methods:
        selected = [by_id[image_id][method] for image_id in ready]
        success = [row for row in selected if row["status"] == "ok"]
        n_ok = len(success)
        n_infeasible = sum(row["status"] == "infeasible" for row in selected)
        n_undecodable = sum(row["status"] == "undecodable" for row in selected)
        if n_ok + n_infeasible + n_undecodable != len(selected):
            raise ValueError("逐图记录出现未知状态")
        successful_counts = (np.asarray([row["counts"] for row in success], dtype=np.int64)
                             if success else np.empty((0, 3), dtype=np.int64))
        success_total = successful_counts.sum(axis=0)
        success_union = int(success_total.sum())
        failure_zero_counts = np.asarray([
            row["counts"] if row["status"] == "ok"
            else [0, 0, int(row["gt_target_pixels"])]
            for row in selected], dtype=np.int64).reshape(-1, 3)
        total = failure_zero_counts.sum(axis=0)
        denominator = int(total.sum())
        score = float(total[0] / denominator) if complete and denominator else None
        interval = None
        if resampling is not None and denominator:
            samples = failure_zero_counts[resampling].sum(axis=1)
            unions = samples.sum(axis=1)
            valid = unions > 0
            if valid.any():
                interval = np.quantile(samples[valid, 0] / unions[valid],
                                       [0.025, 0.975]).tolist()
        methods_out[method] = {
            "iou_with_failures_zero": score, "bootstrap_95_ci": interval,
            "all_image_counts_tp_fp_fn": total.tolist() if complete else None,
            "successful_only_iou": (float(success_total[0] / success_union)
                                    if success_union else None),
            "successful_only_counts_tp_fp_fn": success_total.tolist(),
            "n_ok": n_ok, "n_infeasible": n_infeasible,
            "n_undecodable": n_undecodable,
            "decodable_rate": n_ok / len(ready) if ready else None,
            "infeasible_rate": n_infeasible / len(ready) if ready else None,
            "undecodable_rate": n_undecodable / len(ready) if ready else None,
        }
    return {"task": task, "class_id": TARGETS[task],
            "n_expected_images": len(expected_image_ids),
            "n_complete_images": len(ready), "all_expected_complete": complete,
            "n_paired_successful_images": len(paired),
            "failure_policy": "infeasible/undecodable: TP=0, FP=0, FN=GT target pixel count",
            "methods": methods_out}


def run_figure8a_packet(config: ReplayConfig, device: torch.device, *, task: str,
                        trace_csv: str | Path, trace_provenance: str | Path,
                        limitnet_checkpoint: str | Path, output: str | Path,
                        budget_bytes: int = 30_000, seed: int = 2026,
                        max_images: int | None = None,
                        sender_tuning_report: str | Path | None = None,
                        receiver_checkpoint: str | Path | None = None,
                        background_lowpass_sigma: float | None = None,
                        pilot_codec_candidate: str | None = None,
                        allow_packet_size_mismatch: bool = False,
                        limitnet_mode: str = "official224",
                        receiver_mask_only: bool = False,
                        no_case_capture: bool = False) -> dict:
    """默认要求全 500 张；max_images 仅供烟测，产物必须标作 pilot。"""
    from replay.evaluation.segmentation import CityscapesSegmenter
    from replay.evaluation.figure8a_capture import CAPTURE_PROTOCOL, ExactCaseCapture
    from replay.knowledge.encoders import SharedKB
    from replay.knowledge.store import PosteriorKB
    from replay.receiver.infer import load_receiver, restore
    from replay.evaluation.receiver_mask_only import compose_mask_only
    if task not in TARGETS or budget_bytes <= 0:
        raise ValueError("任务或字节预算无效")
    if receiver_mask_only and receiver_checkpoint is None:
        raise ValueError("mask-only 接收端消融必须显式指定 --receiver-checkpoint")
    if no_case_capture and receiver_checkpoint is None:
        raise ValueError("关闭同期案例捕获仅允许独立适配接收端消融")
    methods = methods_for_limitnet_mode(limitnet_mode)
    limit_method = f"limitnet_{limitnet_mode}"
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images 必须为正；正式实验省略此参数")
    pairs = list(cityscapes_pairs(config.data_path("cityscapes_root"), "val"))
    if len(pairs) != 500:
        raise ValueError(f"Cityscapes val 必须含完整 500 张有标注原图，实际 {len(pairs)}")
    if sender_tuning_report is not None and background_lowpass_sigma is not None:
        raise ValueError("不能同时指定训练集调参报告和手动 sigma")
    if pilot_codec_candidate is not None and (
            max_images is None or sender_tuning_report is not None or
            background_lowpass_sigma is not None):
        raise ValueError("手动 codec candidate 仅供独立 pilot，不能与报告或手动 sigma 共用")
    if max_images is None and sender_tuning_report is None:
        raise ValueError("完整 500 张正式评测必须传入 --sender-tuning-report")
    if receiver_checkpoint is not None and sender_tuning_report is None:
        raise ValueError("适配接收端消融必须提供相同的 TRAIN 发送端调参报告")
    report_path = config.path(sender_tuning_report) if sender_tuning_report is not None else None
    codec_candidate: dict | None = None
    if report_path is not None:
        train_ids = {path.stem for path, _ in
                     cityscapes_pairs(config.data_path("cityscapes_root"), "train")}
        if train_ids.intersection(path.stem for path, _ in pairs):
            raise ValueError("Cityscapes train 与 val 图像 ID 重叠，拒绝训练集调参报告")
        report_protocol = json.loads(report_path.read_text(encoding="utf-8")).get(
            "settings", {}).get("protocol")
        if report_protocol == "figure8a_train_dct_ablation_v1":
            codec_candidate, tuning_sha256 = load_dct_sender_candidate(
                config, report_path, task=task, budget_bytes=budget_bytes,
                train_image_ids=train_ids)
            sigma = float(codec_candidate["sigma"])
            sigma_source = "train_report_multidimensional"
        else:
            sigma, tuning_sha256 = _load_sender_tuning_sigma(
                config, report_path, task=task, budget_bytes=budget_bytes,
                train_image_ids=train_ids)
            sigma_source = "train_report"
    else:
        if pilot_codec_candidate is not None:
            from replay.evaluation.figure8a_dct_ablation import DEFAULT_CANDIDATES
            matches = [item for item in DEFAULT_CANDIDATES
                       if item["id"] == pilot_codec_candidate]
            if len(matches) != 1:
                raise ValueError("pilot codec candidate 未在预注册训练集候选中")
            codec_candidate = matches[0].copy()
            sigma = float(codec_candidate["sigma"])
        else:
            sigma = 0.0 if background_lowpass_sigma is None else float(background_lowpass_sigma)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("pilot 背景低通 sigma 必须是有限非负数")
        tuning_sha256 = None
        sigma_source = ("pilot_candidate" if codec_candidate is not None else
                        "pilot_default" if background_lowpass_sigma is None else
                        "pilot_manual")
    if max_images is not None:
        pairs = pairs[:max_images]
    checkpoint = config.path(limitnet_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少 LimitNet 已训练权重：{checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint_data.get("metadata", {}) if isinstance(checkpoint_data, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    limitnet_source = str(metadata.get("source", "unrecorded"))
    limitnet_variant = _validate_limitnet_checkpoint_metadata(limitnet_mode, metadata)
    recorded_lambda = metadata.get("lambda", metadata.get("lambda_rd"))
    limitnet_lambda_recorded = (recorded_lambda is not None and
                                str(recorded_lambda) in {"0.01", "0.010", "1e-2"})
    del checkpoint_data
    trace = load_packet_trace_csv(
        config.path(trace_csv),
        provenance=load_trace_provenance_json(config.path(trace_provenance)))
    if re.search(r"(?:^|[^a-z0-9])5g(?:$|[^a-z0-9])",
                 trace.provenance.network.casefold()) is None:
        raise ValueError("图 8(a) 本入口仅接受来源明确标注为 5G 的逐包轨迹")
    source_packet_bytes, packet_size_mismatch, trace_interpretation = (
        _trace_packet_size_policy(
            trace.provenance.capture_method, formal=max_images is None,
            allow_mismatch=allow_packet_size_mismatch))
    receiver_path = (config.receiver_checkpoint() if receiver_checkpoint is None
                     else config.path(receiver_checkpoint))
    if not receiver_path.is_file():
        raise FileNotFoundError(f"缺少接收端权重：{receiver_path}")
    adapted_receiver: dict | None = None
    if receiver_checkpoint is not None:
        if receiver_path.resolve() == config.receiver_checkpoint().resolve():
            raise ValueError("适配接收端消融必须使用独立于原始预训练接收端的权重")
        payload = torch.load(receiver_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError("适配接收端 checkpoint 缺少 state_dict")
        report_settings = json.loads(report_path.read_text(encoding="utf-8"))["settings"]
        tuning_ids = {image_id for sampled in report_settings["sampled_image_ids"].values()
                      for image_id in sampled}
        adapted_receiver = _validate_packet_adapted_receiver_metadata(
            payload.get("metadata"), task=task, budget_bytes=budget_bytes,
            trace_sha256=trace.sha256, sender_tuning_sha256=tuning_sha256,
            codec_candidate=codec_candidate, train_image_ids=train_ids,
            tuning_image_ids=tuning_ids,
            initial_receiver_sha256=_digest(config.receiver_checkpoint()),
            source_packet_bytes=source_packet_bytes)
        del payload
    verify_model_lock(config, "segmenter")
    output = config.path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_cfg = config.section("image")
    clip = config.section("models")["clip"]
    settings = {
        "protocol": "figure8a_cityscapes_packet_v9", "task": task,
        "class_id": TARGETS[task], "split": "val", "native_resolution": [2048, 1024],
        "image_ids": [path.stem for path, _ in pairs],
        "full_validation": max_images is None, "pilot": max_images is not None,
        "seed": seed,
        "budget_wire_bytes": budget_bytes,
        "methods": list(methods),
        "limitnet_mode": limitnet_mode,
        "limitnet_codec_input_resolution": ([224, 224] if limitnet_mode == "official224"
                                               else [2048, 1024]),
        "limitnet_output_resize": ("bilinear_to_original_resolution"
                                   if limitnet_mode == "official224" else "none"),
        "focus": "KDC: Cityscapes fine GT oracle ROI; LimitNet: model saliency",
        "trace_sha256": trace.sha256, "trace_provenance": vars(trace.provenance),
        "trace_mapping": "SHA256(seed,image_id,trace_sha256) fixed start; same for all tasks and codecs",
        "trace_reserved_window_packets": TRACE_RESERVED_WINDOW_PACKETS,
        "source_packet_bytes": source_packet_bytes,
        "packet_size_mismatch": packet_size_mismatch,
        "allow_packet_size_mismatch": allow_packet_size_mismatch,
        "trace_interpretation": trace_interpretation,
        "control_delivery": "reliable; data packets lost by observed trace flags",
        "primary_metric": "500-image class IoU; failed frames contribute TP=0, FP=0, FN=GT target pixels",
        "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
        "packet_header_bytes": PACKET_HEADER_BYTES,
        "jpeg_restart_blocks": 2048,
        "jpeg_optimize_huffman": True,
        "kdc_background_lowpass_sigma": sigma,
        "sender_sigma_source": sigma_source,
        "sender_tuning_report": str(report_path) if report_path is not None else None,
        "sender_tuning_report_sha256": tuning_sha256,
        "kdc_fast_prequant": True,
        "kdc_zero_loss_receiver_bypass": True,
        "jpeg_baseline_background_lowpass_sigma": 0.0,
        "background_lowpass_selection": (
            "selected_sigma from validated train-only report with >=50 images per class"
            if report_path is not None else
            "pilot-only default/manual sigma; not eligible for formal 500-image figure"),
        "jpeg_restart_selection": (
            "engineering candidate from 20 seeded Cityscapes train images at native size, "
            "4:4:4 q1 optimize=True with packet+manifest bytes but without KDC frame header: "
            "RST512 3/20, RST1024 12/20, RST2048 13/20, RST4096 14/20 "
            "at 29 kB; full val infeasible rates must be reported separately"),
        "config_sha256": sha256(json.dumps(config.values, sort_keys=True,
                                            default=str).encode("utf-8")).hexdigest(),
        "kb_path": str(config.data_path("kb_path")),
        "limitnet_checkpoint_sha256": _digest(checkpoint),
        "limitnet_source": limitnet_source,
        "limitnet_checkpoint_source": limitnet_source,
        "limitnet_model_variant": limitnet_variant,
        "limitnet_input_mode": metadata.get("input_mode"),
        "limitnet_finetune_pixel_rescaling": metadata.get("pixel_rescaling"),
        "limitnet_training_dataset": metadata.get("training_dataset"),
        "limitnet_epochs_completed": metadata.get("epochs_completed"),
        "limitnet_training_recipe": metadata.get("training_recipe"),
        "limitnet_finetune_loss_weights": metadata.get("loss_weights"),
        "limitnet_finetune_keep_fractions": metadata.get("keep_fractions"),
        "limitnet_recorded_lambda": str(recorded_lambda) if recorded_lambda is not None else None,
        "limitnet_lambda_0_01_recorded": limitnet_lambda_recorded,
        "receiver_checkpoint_sha256": _digest(config.receiver_checkpoint()),
        "segmenter_revision": config.section("models")["segmenter"]["revision"],
        "segmenter_tile": int(image_cfg["segformer_tile"]),
        "segmenter_stride": int(image_cfg["segformer_stride"]),
        "limitations": [
            "JPEG restart/control wire format and LimitNet packetization are replay designs, not author code.",
            "KDC GT oracle focus is offline analysis, not deployable sender inference.",
            "RST2048 and Huffman optimization are shared JPEG/KDC engineering settings, not stated in the paper.",
            "KDC-only background Gaussian sigma is selected from train-only data for formal runs; not specified by the paper.",
            "Zero-loss KDC bypasses the receiver and reuses direct decode; this rule is not explicit in the paper.",
            "KDC fast_prequant vectorizes DCT and can have small numerical differences from the scalar path.",
            "Trace provenance is read from the supplied JSON; original capture must be audited separately.",
            "A source packet size unlike 1200B application fragments yields a measured-loss-pattern stress test, not matched MTU conditions.",
            "LimitNet uses its own saliency; its replay latent packet format still differs from the official implementation.",
            "official224 keeps the official fixed input/output size and bilinearly restores the Cityscapes canvas for the shared segmenter; native_adapted is a separately labeled strong ablation.",
            "Cityscapes LimitNet fine-tuning loss weights and fraction schedule are replay designs because the KDC paper does not disclose this training recipe.",
            "The released LimitNet train.py has no lambda=0.01 switch; this KDC-paper setting cannot be verified from a project fine-tuned checkpoint without an explicit definition.",
            "LimitNet lambda=0.01 cannot be claimed unless it is recorded in checkpoint metadata.",
            "Receiver trained with random missing patches may not match JPEG restart packet loss.",
        ],
    }
    if codec_candidate is not None:
        # 新消融目录与旧 v9/4:4:4 目录严格隔离，防止半途混入不同 JPEG 配置。
        settings.update({
            "protocol": "figure8a_cityscapes_packet_v10_codec_selected",
            "kdc_codec_candidate": codec_candidate,
            "kdc_jpeg_subsampling": codec_candidate["subsampling"],
            "jpeg_baseline_subsampling": codec_candidate["subsampling"],
            "jpeg_restart_blocks": codec_candidate["restart_blocks"],
            "jpeg_baseline_restart_blocks": codec_candidate["restart_blocks"],
            "jpeg_baseline_optimize_huffman": True,
            "jpeg_baseline_codec_match": (
                "same subsampling, restart spatial coverage, Huffman, budget and packet loss; "
                "only task-aware prequantization/lowpass differs"),
            "kdc_roi_halo_pixels": codec_candidate["roi_halo_pixels"],
            "kdc_zero_threshold_multiplier": codec_candidate["zero_threshold_multiplier"],
            "background_lowpass_selection": (
                "full Cityscapes train report selected the complete codec candidate"
                if report_path is not None else "pilot-only registered codec candidate"),
            "jpeg_restart_selection": "same train-selected spatial RST policy for KDC and JPEG",
        })
        settings["limitations"] = [
            item for item in settings["limitations"]
            if not item.startswith("RST2048 and Huffman optimization")
        ] + [
            "JPEG subsampling/RST/Huffman are matched for KDC and JPEG but are replay choices, not specified by the paper.",
            "DCT strength, coefficient-zero threshold, and ROI halo were selected from Cityscapes train only; paper leaves them unspecified.",
        ]
    if adapted_receiver is not None:
        # 与论文式预训练接收端隔离：权重、协议标记和输出目录均不能混用。
        settings.update({
            "protocol": "figure8a_cityscapes_packet_v10_codec_selected_receiver_adapted",
            "receiver_checkpoint_sha256": _digest(receiver_path),
            "receiver_checkpoint_path": str(receiver_path),
            "receiver_checkpoint_source": "Cityscapes TRAIN measured-loss packet adaptation; separate ablation",
            "receiver_adaptation_metadata": adapted_receiver,
        })
        settings["limitations"].append(
            "The packet-adapted receiver is a separately labeled Cityscapes TRAIN ablation, not the paper's original receiver pretraining.")
    if receiver_mask_only:
        # 独立输出协议，防止与完整 RGB 恢复的主报告或既有适配消融混用。
        settings["protocol"] += "_mask_only"
        settings["receiver_output_policy"] = "replace_missing_entropy_intervals_only_no_halo"
        settings["limitations"].append(
            "Mask-only composition is a replay ablation; preservation outside the entropy mask assumes no JPEG corruption propagates there.")
    if no_case_capture:
        # 预填的主接收端记录没有同一适配权重的同期图像，必须显式关闭捕获。
        settings.update({"case_capture": "disabled_explicit_adapted_ablation",
                         "capture_protocol": None, "capture_manifest": None})
        seed_path = output / "seed_provenance.json"
        if seed_path.is_file():
            from replay.evaluation.figure8a_seed import validate_seed_for_settings
            validate_seed_for_settings(output, settings)
            settings["seed_provenance"] = "seed_provenance.json"
            settings["seed_provenance_sha256"] = _digest(seed_path)
        elif (output / "runs.jsonl").exists() and not (output / "settings.json").exists():
            raise ValueError("关闭案例捕获的预填目录缺少 seed_provenance.json")
    else:
        settings["capture_protocol"] = CAPTURE_PROTOCOL
        settings["capture_manifest"] = "captured_cases/manifest.json"
    settings_path = output / "settings.json"
    capture_enabled = not no_case_capture
    if settings_path.exists():
        previous_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous_settings != settings:
            legacy_settings = {key: value for key, value in settings.items()
                               if key not in {"capture_protocol", "capture_manifest", "seed"}}
            if codec_candidate is None and previous_settings == legacy_settings:
                # 旧 v9 输出目录只允许原样续跑；不要半途宣称有同期案例捕获。
                settings = previous_settings
                capture_enabled = False
            else:
                raise ValueError("输出目录已有不同实验设置；拒绝混合续跑")
    else:
        write_json(settings_path, settings)
    records_path = output / "runs.jsonl"
    completed: dict[tuple[str, str], dict] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = row["image_id"], row["method"]
                if key in completed:
                    raise ValueError(f"重复逐图记录：{key}")
                completed[key] = row
    capture = (ExactCaseCapture(output, settings, existing_rows=len(completed))
               if capture_enabled else None)

    shared = SharedKB.from_config(config, device)
    kb = PosteriorKB.load(config.data_path("kb_path"))
    receiver = (load_receiver(config, device) if receiver_checkpoint is None else
                load_receiver(config, device, checkpoint_path=receiver_path))
    segmenter = CityscapesSegmenter(config.model_path("segmenter"), device,
                                     int(image_cfg["segformer_tile"]),
                                     int(image_cfg["segformer_stride"]))
    limitnet = load_limitnet(checkpoint, device)
    session_bytes = SessionHeader(task, (task,), clip["repo"], clip["revision"]).to_bytes()
    session_sent = any(row["method"] == "kdc_direct" and row.get("sent_bytes") is not None
                       and row.get("session_header_bytes", 0) > 0
                       for row in completed.values())

    def append(row: dict) -> None:
        key = row["image_id"], row["method"]
        if key in completed:
            return
        with records_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        completed[key] = row

    def report() -> dict:
        result = {"settings": settings,
                  "summary": _summarize(list(completed.values()), task, seed,
                                        settings["image_ids"], methods),
                  "runs": list(completed.values())}
        write_json(output / "report.json", result)
        return result

    for number, (image_path, label_path) in enumerate(pairs, 1):
        image_id = image_path.stem
        if all((image_id, method) in completed for method in methods):
            continue
        image, truth = load_cityscapes_pair(image_path, label_path)
        if image.size != (2048, 1024):
            raise ValueError(f"非原生 Cityscapes 分辨率：{image_path} {image.size}")
        exact_pictures: dict[str, Image.Image | None] = {
            "jpeg": None, "limitnet": None, "kdc_direct": None, "kdc_restored": None}
        exact_predictions: dict[str, np.ndarray | None] = {
            name: None for name in exact_pictures}
        exact_missing = np.zeros((image.height, image.width), dtype=np.float32)
        focus = ground_truth_patches(truth, TARGETS[task], int(image_cfg["patch_size"]))
        previous_kdc = completed.get((image_id, "kdc_direct"))
        if previous_kdc is not None and previous_kdc.get("sent_bytes") is not None:
            session = (session_bytes if previous_kdc.get("session_header_bytes", 0) > 0
                       else None)
        else:
            session = None if session_sent else session_bytes
        frame = FrameHeader(image.width, image.height,
                            int(image_cfg["patch_size"]), focus).to_bytes()
        coded: dict[str, tuple[object, bytes | None, dict] | str] = {}
        for method in ("jpeg", "kdc", "limitnet"):
            try:
                if method == "jpeg":
                    wire, payload, info = _jpeg_wire(image, focus=None, session=None,
                                                       frame=b"", budget=budget_bytes,
                                                       config=config,
                                                       codec_candidate=codec_candidate)
                elif method == "kdc":
                    wire, payload, info = _jpeg_wire(image, focus=focus, session=session,
                                                       frame=frame, budget=budget_bytes,
                                                       config=config,
                                                       background_lowpass_sigma=sigma,
                                                       codec_candidate=codec_candidate)
                else:
                    limit_input = _limitnet_input(image, limitnet_mode)
                    wire, received_layers = _limit_under_budget(
                        limitnet, limit_input, None, budget_bytes)
                    payload = None
                    info = {"layers_sent": received_layers, "source": limitnet.source,
                            "focus_source": "model_saliency",
                            "codec_input_size": list(limit_input.size)}
                coded[method] = wire, payload, info
            except ValueError as error:
                if not _is_budget_error(error):
                    raise
                coded[method] = str(error)
        packet_counts = {name: len(value[0].packets) for name, value in coded.items()
                         if not isinstance(value, str)}
        max_packets = max(packet_counts.values(), default=0)
        if max_packets > TRACE_RESERVED_WINDOW_PACKETS:
            raise ValueError(f"当前图需 {max_packets} 包，超过固定轨迹窗口 "
                             f"{TRACE_RESERVED_WINDOW_PACKETS} 包")
        trace_start = _fixed_trace_start(trace, seed=seed, image_id=image_id)
        loss_by_method = {
            name: trace.select_segment(n_packets=count,
                                       window_packets=TRACE_RESERVED_WINDOW_PACKETS,
                                       seed=seed, case_id=image_id,
                                       start_index=trace_start)
            for name, count in packet_counts.items() if count > 0
        }
        base = {"image_id": image_id, "source_image": str(image_path),
                "task": task, "class_id": TARGETS[task],
                "budget_wire_bytes": budget_bytes, "focus_patches": int(focus.sum()),
                "gt_target_pixels": int((truth == TARGETS[task]).sum()),
                "trace_sha256": trace.sha256,
                "trace_start_index": trace_start,
                "trace_window_packets": TRACE_RESERVED_WINDOW_PACKETS,
                "trace_used_packets_max": max_packets}
        for name, emitted_methods in (("jpeg", ("jpeg",)),
                                      ("kdc", ("kdc_direct", "kdc_restored")),
                                      ("limitnet", (limit_method,))):
            if all((image_id, method) in completed for method in emitted_methods):
                continue
            value = coded[name]
            if isinstance(value, str):
                for method in emitted_methods:
                    append({**base, "method": method, "status": "infeasible",
                            "reason": value, "counts": None, "sent_bytes": None})
                continue
            wire, payload, info = value
            replay = loss_by_method.get(name)
            if name == "limitnet":
                if replay is None:
                    raise AssertionError("LimitNet 已编码至少一层，但没有对应的数据分片")
                loss = replay.loss_trace
                trace_meta = {"packet_count": len(loss),
                              "lost_packets": sum(loss),
                              "loss_rate": replay.loss_rate}
                candidate, accepted, received_bytes = _receive_limit(wire, loss, limitnet)
                fields = {"sent_bytes": wire.sent_bytes,
                          "received_bytes": received_bytes,
                          "wire_sha256": _wire_sha256(wire),
                          "layers_sent": info["layers_sent"],
                          "layers_received": accepted,
                          "limitnet_mode": limitnet_mode,
                          "codec_input_size": info["codec_input_size"],
                          "limitnet_source": info["source"],
                          "channel_kind": "measured_trace_on_limitnet_layer_packets",
                          **trace_meta}
                if accepted == 0:
                    append({**base, "method": limit_method, "status": "undecodable",
                            "reason": "没有连续接收到任何完整潜变量层",
                            "counts": None, **fields})
                    continue
                if candidate.size != tuple(info["codec_input_size"]):
                    raise AssertionError("LimitNet 解码尺寸与编码输入不一致")
                candidate = _limitnet_output(candidate, image.size, limitnet_mode)
                if candidate.size != image.size:
                    raise AssertionError("LimitNet 解码改变了原图尺寸")
                fields["decoded_image_sha256"] = _image_sha256(candidate)
                prediction = segmenter.predict(candidate)
                exact_pictures["limitnet"] = candidate
                exact_predictions["limitnet"] = prediction
                append({**base, "method": limit_method, "status": "ok",
                        "counts": list(binary_counts(prediction, truth, TARGETS[task])),
                        **fields})
                continue
            if replay is None or payload is None:
                raise AssertionError("JPEG 封包缺少实测丢包事件")
            candidate, missing, fields = _decode_jpeg_loss(
                wire, payload, replay.loss_trace, budget=budget_bytes,
                allow_420=bool(codec_candidate is not None and
                               codec_candidate["subsampling"] == 2))
            fields["wire_sha256"] = _wire_sha256(wire)
            fields["jpeg_quality"] = info["jpeg_quality"]
            if codec_candidate is not None:
                fields["jpeg_subsampling"] = codec_candidate["subsampling"]
                fields["restart_blocks"] = codec_candidate["restart_blocks"]
                fields["codec_candidate_id"] = codec_candidate["id"]
            if name == "kdc":
                fields["session_header_bytes"] = len(session or b"")
                fields["frame_header_bytes"] = len(frame)
                session_sent = True
            if candidate is None:
                for method in emitted_methods:
                    append({**base, "method": method, "status": "undecodable",
                            "counts": None, **fields})
                continue
            if candidate.size != image.size:
                raise AssertionError("JPEG 丢包解码改变了原图尺寸")
            fields["decoded_image_sha256"] = _image_sha256(candidate)
            if name == "jpeg":
                prediction = segmenter.predict(candidate)
                exact_pictures["jpeg"] = candidate
                exact_predictions["jpeg"] = prediction
                append({**base, "method": "jpeg", "status": "ok",
                        "counts": list(binary_counts(prediction, truth, TARGETS[task])),
                        **fields})
            else:
                direct_prediction = segmenter.predict(candidate)
                direct_counts = list(binary_counts(direct_prediction, truth, TARGETS[task]))
                exact_pictures["kdc_direct"] = candidate
                exact_predictions["kdc_direct"] = direct_prediction
                exact_missing = missing
                append({**base, "method": "kdc_direct", "status": "ok",
                        "counts": direct_counts, "receiver_applied": False, **fields})
                restored_hashes: list[str] = []
                def repaired_counts() -> list[int]:
                    restored = restore(candidate, [task], shared, kb, receiver,
                                       int(config.section("knowledge")["rx_top_k"]), device,
                                       update_session=False,
                                       merge_cosine=float(config.section("knowledge")["session_merge_cosine"]),
                                       missing_pixels=missing)
                    if receiver_mask_only:
                        restored = compose_mask_only(candidate, restored, missing)
                    restored_hashes.append(_image_sha256(restored))
                    prediction = segmenter.predict(restored)
                    exact_pictures["kdc_restored"] = restored
                    exact_predictions["kdc_restored"] = prediction
                    return list(binary_counts(prediction, truth, TARGETS[task]))

                restored_counts, receiver_applied = _kdc_restored_counts(
                    direct_counts, fields["lost_intervals"], missing, repaired_counts)
                if not receiver_applied:
                    exact_pictures["kdc_restored"] = candidate
                    exact_predictions["kdc_restored"] = direct_prediction
                append({**base, "method": "kdc_restored", "status": "ok",
                        "counts": restored_counts, "receiver_applied": receiver_applied,
                        **({"receiver_output_policy": "mask_only_no_halo"}
                           if receiver_mask_only else {}),
                        "restored_image_sha256": (restored_hashes[0] if receiver_applied
                                                  else fields["decoded_image_sha256"]),
                        "receiver_bypass_reason": (None if receiver_applied else
                                                   "no_missing_entropy_interval"), **fields})
        if capture is not None:
            capture.consider(
                image_id=image_id, original=image, truth=truth,
                missing=exact_missing, pictures=exact_pictures,
                predictions=exact_predictions,
                rows={method: completed[(image_id, method)] for method in methods})
        image.close()
        if number % 10 == 0 or number == len(pairs):
            report()
            print(f"[Fig8(a) {task} {number}/{len(pairs)}] rows={len(completed)}", flush=True)
    return report()
