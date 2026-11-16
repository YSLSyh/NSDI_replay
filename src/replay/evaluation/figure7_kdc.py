"""ImageNet-1K 图 7 的 KDC 图像编解码适配器。

发送端只读取原图、离线任务描述和语义桶，不读取分类真值。逐图的会话头、
重要性掩码、JPEG 结构、RST 区间表及数据包头都计入传输预算。
分类任务语义桶或可用训练参考 KB 不存在时明确拒绝运行，不能沿用
Cityscapes 的自动驾驶桶，也不能读取 ImageNet val 真值来决定重要区域。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from replay.core.config import ReplayConfig, load_config
from replay.transmission.header import FrameHeader, SessionHeader
from replay.transmission.jpeg import encode_under_budget
from replay.transmission.packet_loss import plan_entropy_packets
from replay.transmission.packet_wire import (
    CONTROL_FIXED_BYTES, INTERVAL_MANIFEST_BYTES, MIN_PACKET_HEADER_BYTES,
    PacketizedFrame, pack_packetized_frame, receive_packetized_frame,
)
from replay.transmission import packet_wire as _wire_schema


PACKET_PAYLOAD_BYTES = 1200
PACKET_HEADER_BYTES = MIN_PACKET_HEADER_BYTES


@dataclass(frozen=True)
class Figure7KdcSpec:
    config_path: Path
    task_description: str
    bins_cache: Path
    kb_path: Path
    receiver_checkpoint: Path
    fixed_loss_budget_bytes: int
    update_session: bool = False


def _classification_config(spec: Figure7KdcSpec) -> ReplayConfig:
    base = load_config(spec.config_path)
    values = deepcopy(base.values)
    values["task"]["description"] = spec.task_description
    values["task"]["bin_cache"] = str(spec.bins_cache)
    # 自动驾驶的四个必选类别和 Cityscapes 词表不能约束 ImageNet 分类任务。
    for key in ("required_bins", "allowed_bins", "bin_aliases", "taxonomy"):
        values["task"].pop(key, None)
    values["data"]["kb_path"] = str(spec.kb_path)
    values["data"]["checkpoint"] = str(spec.receiver_checkpoint)
    return ReplayConfig(base.root, values)


def missing_kdc_classification_assets(spec: Figure7KdcSpec) -> list[str]:
    """轻量依赖审计；允许复用 COCO/FSS 等 train-only 图像参考。"""
    missing: list[str] = []
    if not spec.task_description.strip() or spec.task_description.strip().lower() == "autonomous driving":
        missing.append("需要独立的图像分类任务描述，不能复用自动驾驶任务")
    if spec.fixed_loss_budget_bytes <= 0:
        missing.append("图 7(a) 需要显式指定固定线字节预算")
    if not Path(spec.config_path).is_file():
        return missing + [f"项目配置不存在：{spec.config_path}"]
    config = _classification_config(spec)
    lock = config.path(config.values["models_lock_file"])
    if not lock.is_file():
        missing.append(f"模型版本锁不存在：{lock}")
    for name in ("clip", "clipseg"):
        path = config.model_path(name)
        if not path.is_dir() or not (path / "config.json").is_file():
            missing.append(f"SharedKB/CLIPSeg 权重缺失：{path}")
    checkpoint = config.path(spec.receiver_checkpoint)
    if not checkpoint.is_file():
        missing.append(f"图像恢复权重缺失：{checkpoint}")
    bins_path = config.path(spec.bins_cache)
    if not bins_path.is_file():
        missing.append(f"图像分类任务的 4–8 类语义桶缓存缺失：{bins_path}")
    else:
        try:
            cache = json.loads(bins_path.read_text(encoding="utf-8"))
            bins = cache.get("bins")
            if (cache.get("task") != spec.task_description or
                    not isinstance(bins, list) or not 4 <= len(bins) <= 8 or
                    not all(isinstance(item, str) and item.strip() for item in bins) or
                    len(set(bins)) != len(bins)):
                missing.append("分类语义桶缓存的任务、类别数量或类别名称无效")
            else:
                audit = cache.get("selection_audit")
                attempts = audit.get("attempts") if isinstance(audit, dict) else None
                selected_index = audit.get("selected_attempt_index") if isinstance(audit, dict) else None
                if selected_index is None and isinstance(attempts, list):
                    selected_index = len(attempts) - 1
                selected = (attempts[selected_index]
                            if isinstance(attempts, list) and attempts and
                            type(selected_index) is int and 0 <= selected_index < len(attempts)
                            else None)
                if (cache.get("semantic_diversity_verified") is not True or
                  cache.get("semantic_distinct_count") != len(bins) or
                  cache.get("broad_family_verified") is not True or
                  not isinstance(selected, dict) or selected.get("accepted") is not True or
                  selected.get("bins") != bins or
                  selected.get("family_audit", {}).get("accepted") is not True):
                    missing.append("分类语义桶缺少上位对象族/同义过滤审计；旧缓存需重跑")
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            missing.append("分类语义桶缓存不是有效 JSON")
    kb_path = config.path(spec.kb_path)
    for filename in ("meta.json", "records.json", "text.faiss", "data_vectors.npy"):
        if not (kb_path / filename).is_file():
            missing.append(f"分类 PosteriorKB 缺少 {filename}：{kb_path}")
    records_file = kb_path / "records.json"
    if records_file.is_file():
        try:
            records = json.loads(records_file.read_text(encoding="utf-8"))
            if (not isinstance(records, list) or not records or
                    any(not isinstance(record, dict) or
                        record.get("split") != "train" for record in records)):
                missing.append("PosteriorKB 必须只包含训练集记录，不得有 val/session")
            elif not any(record.get("has_data") is True for record in records):
                missing.append("PosteriorKB 缺少可供分类任务检索的训练图像嵌入")
        except (OSError, ValueError, AttributeError):
            missing.append("分类 PosteriorKB 记录文件无效")
    return missing


class KdcWireBytes(bytes):
    """完整实际线字节，附带本地仿真所需的分包边界。"""

    def __new__(cls, frame: PacketizedFrame):
        instance = super().__new__(cls, frame.control + b"".join(frame.packets))
        instance.frame = frame
        return instance


def _missing_mask_from_control(
    control: bytes, lost_intervals: tuple[tuple[int, int], ...], *,
    width: int, height: int,
) -> np.ndarray:
    """仅从接收端可靠控制清单重建缺失 MCU，绝不读取发送端原始 JPEG。"""
    prefix = _wire_schema._PREFIX
    interval_format = _wire_schema._INTERVAL
    (_, _, _, session_length, frame_length, _, interval_count, _, _, _) = (
        prefix.unpack_from(control))
    start = prefix.size + session_length + frame_length
    missing = np.zeros((height, width), dtype=np.float32)
    lost = set(lost_intervals)
    mcus_per_row = math.ceil(width / 8)  # 该适配器固定 JPEG 4:4:4。
    for ordinal in range(interval_count):
        _, _, scan, interval, mcu_start, mcu_end = interval_format.unpack_from(
            control, start + ordinal * interval_format.size)
        if (scan, interval) not in lost:
            continue
        for mcu in range(mcu_start, mcu_end):
            row, col = divmod(mcu, mcus_per_row)
            missing[row * 8:min((row + 1) * 8, height),
                    col * 8:min((col + 1) * 8, width)] = 1.0
    return missing


class KdcClassificationCodec:
    """与 figure7_classification.evaluate_codec_point 的协议相容。"""

    name = "kdc"

    def __init__(
        self, *, config: ReplayConfig, bins: list[str], shared: Any,
        importance: Any, kb: Any, receiver: Any, device: Any,
        fixed_loss_budget_bytes: int, update_session: bool = False,
        restorer: Callable[..., Image.Image] | None = None,
    ):
        if not 4 <= len(bins) <= 8 or len(set(bins)) != len(bins):
            raise ValueError("图 7 分类任务语义桶必须是 4–8 个不同类别")
        if fixed_loss_budget_bytes <= 0:
            raise ValueError("图 7(a) 固定线字节预算必须为正")
        self.config = config
        self.bins = bins
        self.shared = shared
        self.importance = importance
        self.kb = kb
        self.receiver = receiver
        self.device = device
        self.fixed_loss_budget_bytes = fixed_loss_budget_bytes
        self.update_session = update_session
        if restorer is None:
            from replay.receiver.infer import restore
            restorer = restore
        self.restorer = restorer

    @classmethod
    def from_spec(cls, spec: Figure7KdcSpec, device: Any) -> "KdcClassificationCodec":
        missing = missing_kdc_classification_assets(spec)
        if missing:
            raise FileNotFoundError("图 7 KDC 分类依赖不完整：\n- " + "\n- ".join(missing))
        from replay.knowledge.bins import read_bins
        from replay.knowledge.encoders import SharedKB
        from replay.knowledge.store import PosteriorKB
        from replay.receiver.infer import load_receiver
        from replay.transmission.importance import RegionImportance

        config = _classification_config(spec)
        bins = read_bins(config.path(spec.bins_cache), spec.task_description)
        shared = SharedKB.from_config(config, device)
        importance = RegionImportance(config, device, shared)
        kb = PosteriorKB.load(config.path(spec.kb_path))
        receiver = load_receiver(config, device,
                                 checkpoint_path=config.path(spec.receiver_checkpoint))
        return cls(config=config, bins=bins, shared=shared,
                   importance=importance, kb=kb, receiver=receiver,
                   device=device, fixed_loss_budget_bytes=spec.fixed_loss_budget_bytes,
                   update_session=spec.update_session)

    def encode(self, image: Image.Image, *, budget_bytes: int | None) -> tuple[KdcWireBytes, dict]:
        budget = self.fixed_loss_budget_bytes if budget_bytes is None else budget_bytes
        if type(budget) is not int or budget <= 0:
            raise ValueError("KDC 线字节预算必须为正整数")
        rgb = image.convert("RGB")
        _, selected, scores = self.importance.map(rgb, self.bins, return_scores=True)
        image_cfg = self.config.section("image")
        importance_cfg = self.config.section("importance")
        patch_size = int(image_cfg["patch_size"])
        frame = FrameHeader(rgb.width, rgb.height, patch_size, selected).to_bytes(compact=True)
        clip = self.config.section("models")["clip"]
        session = SessionHeader(self.config.section("task")["description"],
                                tuple(self.bins), clip["repo"], clip["revision"]).to_bytes()
        jpeg_budget = budget - CONTROL_FIXED_BYTES - len(session) - len(frame)
        if jpeg_budget <= 0:
            raise ValueError("KDC 控制头和语义 mask 已耗尽传输预算")
        jpeg, info = encode_under_budget(
            rgb, scores, patch_size, jpeg_budget, 0,
            int(importance_cfg["dct_high_frequency_cutoff"]),
            luma_strength=float(importance_cfg["prequant_luma_strength"]),
            chroma_strength=float(importance_cfg["prequant_chroma_strength"]),
            gamma=float(importance_cfg["prequant_gamma"]),
            importance_ratio_low=float(importance_cfg["importance_ratio_low"]),
            importance_ratio_high=float(importance_cfg["importance_ratio_high"]),
            fast_prequant=True, restart_marker_rows=1,
            packet_header_bytes=PACKET_HEADER_BYTES,
            packet_manifest_bytes_per_interval=INTERVAL_MANIFEST_BYTES,
            optimize_huffman=True)
        width, height, intervals, packets = plan_entropy_packets(
            jpeg, PACKET_PAYLOAD_BYTES)
        if (width, height) != rgb.size:
            raise AssertionError("KDC 图像分辨率被编码器更改")
        wire = pack_packetized_frame(
            session, frame, jpeg, intervals, packets,
            packet_payload_bytes=PACKET_PAYLOAD_BYTES,
            packet_header_bytes=PACKET_HEADER_BYTES)
        if wire.sent_bytes > budget or wire.sent_bytes != (
                CONTROL_FIXED_BYTES + len(session) + len(frame) + info["codec_wire_bytes"]):
            raise AssertionError("KDC 实际线字节与预算搜索不一致")
        encoded = KdcWireBytes(wire)
        if len(encoded) != wire.sent_bytes:
            raise AssertionError("KDC 线字节封装长度错误")
        return encoded, {
            "wire_bytes": wire.sent_bytes, "loss_packet_count": len(wire.packets),
            "source_codec_bytes": len(jpeg), "jpeg_quality": info["jpeg_quality"],
            "session_header_bytes": len(session), "frame_header_bytes": len(frame),
            "control_bytes": len(wire.control),
            "selected_patch_fraction": float(np.asarray(selected).mean()),
            "budget_bytes": budget, "budget_policy": "per_budget_reencode",
            "control_delivery": "reliable", "jpeg_subsampling": "4:4:4",
            "restart_marker_rows": 1, "packet_payload_bytes": PACKET_PAYLOAD_BYTES,
            "packet_header_bytes": PACKET_HEADER_BYTES,
            "wire_format": "replay_rst_jpeg_packetization_v2",
            "receiver_session_update": self.update_session,
        }

    def decode(
        self, payload: bytes, *, loss_flags: tuple[bool, ...],
        received_prefix_bytes: int | None,
    ) -> Image.Image | None:
        if not isinstance(payload, KdcWireBytes):
            raise TypeError("KDC 解码需要带分包边界的实际线字节对象")
        wire = payload.frame
        if bytes(payload) != wire.control + b"".join(wire.packets):
            raise ValueError("KDC 实际线字节与分包对象不一致")
        if received_prefix_bytes is not None and received_prefix_bytes != len(payload):
            raise ValueError("KDC 已按预算重新编码，不能假装支持原码流任意前缀解码")
        if len(loss_flags) != len(wire.packets):
            raise ValueError("KDC 丢包事件数与熵分片数不一致")
        received = receive_packetized_frame(
            wire.control, [packet for packet, lost in zip(wire.packets, loss_flags)
                           if not lost])
        session = SessionHeader.from_bytes(received.session_header or b"")
        clip = self.config.section("models")["clip"]
        session.verify_model(clip["repo"], clip["revision"])
        if (session.task != self.config.section("task")["description"] or
                session.bins != tuple(self.bins)):
            raise ValueError("KDC 接收会话语义与发送配置不一致")
        frame = FrameHeader.from_bytes(received.frame_header)
        missing = _missing_mask_from_control(
            wire.control, received.lost_intervals,
            width=frame.width, height=frame.height)
        try:
            with Image.open(BytesIO(received.jpeg_payload)) as loaded:
                loaded.load()
                decoded = loaded.convert("RGB")
            if decoded.size != (frame.width, frame.height):
                return None
        except (OSError, ValueError):
            return None
        if not np.any(missing):
            return decoded
        return self.restorer(
            decoded, self.bins, self.shared, self.kb, self.receiver,
            int(self.config.section("knowledge")["rx_top_k"]), self.device,
            update_session=self.update_session,
            merge_cosine=float(self.config.section("knowledge")["session_merge_cosine"]),
            missing_pixels=missing)
