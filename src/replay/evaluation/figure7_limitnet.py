"""图 7 的官方 ImageNet LimitNet 权重及真实字节分层传输适配。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from replay.baselines.limitnet import LimitNetCodec, load_limitnet
from replay.evaluation.figure8a_packet import (
    _limit_wire, _limitnet_input, _limitnet_output, _receive_limit,
)


OFFICIAL_IMAGENET_MD5 = "5cf7bfff45aa5ec8c688e77936d49794"


def _hash(path: Path, kind: str) -> str:
    digest = md5() if kind == "md5" else sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Figure7LimitNetSpec:
    official_raw_checkpoint: Path
    converted_checkpoint: Path
    fixed_loss_budget_bytes: int
    layers: int = 20


def verify_official_imagenet_limitnet(spec: Figure7LimitNetSpec) -> dict[str, Any]:
    """只接受官方 ImageNet 发布权重，不把 Cityscapes 微调权重混入图 7。"""
    raw = Path(spec.official_raw_checkpoint)
    converted = Path(spec.converted_checkpoint)
    if not raw.is_file():
        raise FileNotFoundError(f"缺少官方 LimitNet-ImageNet 原始权重：{raw}")
    if not converted.is_file():
        raise FileNotFoundError(
            f"缺少安全转换后的官方 ImageNet state_dict：{converted}；"
            "先运行 scripts/convert_limitnet_official.py")
    if spec.fixed_loss_budget_bytes <= 0 or not 1 <= spec.layers <= 100:
        raise ValueError("LimitNet 固定线字节预算或层数无效")
    actual_md5 = _hash(raw, "md5")
    if actual_md5 != OFFICIAL_IMAGENET_MD5:
        raise ValueError(f"LimitNet 官方 ImageNet MD5 不符：{actual_md5}")
    raw_sha = _hash(raw, "sha256")
    value = torch.load(converted, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
        raise ValueError("LimitNet 转换权重缺少来源元数据")
    metadata = value["metadata"]
    if (metadata.get("source") != "official_limitnet_imagenet_pretrained" or
            metadata.get("official_checkpoint_sha256") != raw_sha or
            metadata.get("conversion") != "full_pickle_to_pure_state_dict"):
        raise ValueError("LimitNet 转换权重不是已验签的官方 ImageNet 权重")
    return {"official_raw_md5": actual_md5, "official_raw_sha256": raw_sha,
            "converted_sha256": _hash(converted, "sha256"),
            "source": metadata["source"], "layers": spec.layers,
            "paper_lambda_0_01_verified": False,
            "wire_format_author_specified": False}


class LimitNetWireBytes(bytes):
    """原样保存可传输控制清单和潜变量层分片。"""

    def __new__(cls, wire: Any | None, original_size: tuple[int, int]):
        data = b"" if wire is None else wire.control + b"".join(wire.packets)
        instance = super().__new__(cls, data)
        instance.wire = wire
        instance.original_size = original_size
        return instance


class LimitNetClassificationCodec:
    """固定 224 输入、官方 ImageNet 模型显著性、预算内完整潜变量层。"""

    name = "limitnet"

    def __init__(self, codec: LimitNetCodec, *, fixed_loss_budget_bytes: int,
                 provenance: dict[str, Any]):
        if fixed_loss_budget_bytes <= 0:
            raise ValueError("图 7(a) LimitNet 固定线字节预算必须为正")
        self.codec = codec
        self.fixed_loss_budget_bytes = fixed_loss_budget_bytes
        self.provenance = provenance

    @classmethod
    def from_spec(cls, spec: Figure7LimitNetSpec, device: torch.device
                  ) -> "LimitNetClassificationCodec":
        provenance = verify_official_imagenet_limitnet(spec)
        codec = load_limitnet(spec.converted_checkpoint, device, layers=spec.layers)
        if codec.source != "official_limitnet_imagenet_pretrained":
            raise ValueError("LimitNet 加载的模型来源与官方 ImageNet 权重不一致")
        return cls(codec, fixed_loss_budget_bytes=spec.fixed_loss_budget_bytes,
                   provenance=provenance)

    def encode(self, image: Image.Image, *, budget_bytes: int | None
               ) -> tuple[LimitNetWireBytes, dict]:
        budget = self.fixed_loss_budget_bytes if budget_bytes is None else budget_bytes
        if type(budget) is not int or budget <= 0:
            raise ValueError("LimitNet 线字节预算必须为正整数")
        input_image = _limitnet_input(image, "official224")
        stream = self.codec.encode_full(input_image, focus_pixels=None)
        selected = None
        selected_layers = 0
        for count in range(len(stream.layers) + 1):
            candidate = _limit_wire(stream, count, parity_packets_per_layer=0)
            if candidate.sent_bytes > budget:
                break
            selected, selected_layers = candidate, count
        if selected is None:
            payload = LimitNetWireBytes(None, image.size)
            return payload, {"wire_bytes": 0, "loss_packet_count": 0,
                             "source_codec_bytes": 0, "layers_sent": 0,
                             "budget_bytes": budget, "status": "infeasible_header"}
        payload = LimitNetWireBytes(selected, image.size)
        if len(payload) != selected.sent_bytes or len(payload) > budget:
            raise AssertionError("LimitNet 实际线字节与预算不一致")
        return payload, {
            "wire_bytes": selected.sent_bytes,
            "loss_packet_count": len(selected.packets),
            "source_codec_bytes": len(stream.to_bytes(selected_layers)),
            "full_source_codec_bytes": len(stream.to_bytes()),
            "layers_sent": selected_layers, "total_layers": len(stream.layers),
            "budget_bytes": budget, "budget_policy": "complete_layer_prefix",
            "official_input_size": [224, 224], "control_delivery": "reliable",
            "parity_packets_per_layer": 0,
            "wire_format": "replay_limitnet_layer_packetization_v1",
        }

    def decode(self, payload: bytes, *, loss_flags: tuple[bool, ...],
               received_prefix_bytes: int | None) -> Image.Image | None:
        if not isinstance(payload, LimitNetWireBytes):
            raise TypeError("LimitNet 解码需要带分层边界的实际线字节对象")
        wire = payload.wire
        if wire is None:
            if payload or loss_flags:
                raise ValueError("LimitNet 空预算码流状态不一致")
            return None
        if bytes(payload) != wire.control + b"".join(wire.packets):
            raise ValueError("LimitNet 实际线字节与分层封包不一致")
        if received_prefix_bytes is not None and received_prefix_bytes != len(payload):
            raise ValueError("LimitNet 已选完整预算前缀，不支持任意字节截断")
        if len(loss_flags) != len(wire.packets):
            raise ValueError("LimitNet 丢包事件数与分片数不一致")
        try:
            decoded, _, _ = _receive_limit(wire, loss_flags, self.codec)
        except (OSError, ValueError):
            return None
        return _limitnet_output(decoded, payload.original_size, "official224")
