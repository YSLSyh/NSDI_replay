"""LimitNet 渐进码流的真实字节预算、原尺寸恢复与权重来源。"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from replay.baselines.limitnet import LimitNetCodec, LimitNetCore, load_limitnet
from replay.baselines.limitnet import LimitNetStream
from replay.evaluation.figure8a_packet import (_limit_under_budget, _limit_wire,
                                               _receive_limit)


def _sample_image() -> Image.Image:
    pixels = np.random.default_rng(12).integers(0, 256, (34, 49, 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


def test_progressive_stream_counts_real_prefix_bytes_and_preserves_size():
    codec = LimitNetCodec(LimitNetCore(), torch.device("cpu"), layers=5,
                          source="test_weights")
    image = _sample_image()
    stream = codec.encode_full(image)
    sizes = stream.prefix_sizes()
    assert len(sizes) == 5
    assert all(a < b for a, b in zip(sizes, sizes[1:]))
    assert sizes[-1] == len(stream.to_bytes())
    assert codec.decode_prefix(stream, 3).size == image.size
    result = codec.encode_decode(image, target_bytes=sizes[1])
    assert result.layers_received == 2
    assert result.total_bytes == sizes[1]
    assert result.image.size == image.size


def test_oracle_focus_and_header_only_budget(tmp_path):
    core = LimitNetCore()
    checkpoint = tmp_path / "limitnet.pt"
    torch.save({"state_dict": core.state_dict(),
                "metadata": {"source": "cityscapes_finetuned_from_official"}}, checkpoint)
    codec = load_limitnet(checkpoint, torch.device("cpu"), layers=3)
    image = _sample_image()
    focus = np.zeros((image.height, image.width), np.float32)
    focus[5:20, 8:25] = 1
    stream = codec.encode_full(image, focus_pixels=focus)
    result = codec.encode_decode(image, target_bytes=len(stream.header), focus_pixels=focus)
    assert result.total_bytes == len(stream.header)
    assert result.layers_received == 0
    assert result.image.size == image.size
    assert result.source == "cityscapes_finetuned_from_official+oracle_roi"
    try:
        codec.encode_decode(image, target_bytes=len(stream.header) - 1)
    except ValueError as error:
        assert "预算" in str(error)
    else:
        raise AssertionError("低于码流头的预算必须失败")


def test_packet_loss_keeps_independent_later_layers():
    class Collector:
        def decode_bytes(self, payload: bytes) -> bytes:
            return payload

    stream = LimitNetStream(b"header", (b"A" * 1500, b"B" * 500, b"C" * 700))
    wire = _limit_wire(stream, 3)
    assert len(wire.packets) == 4
    decoded, accepted, received_bytes = _receive_limit(
        wire, (False, True, False, False), Collector())
    assert decoded == b"header" + b"B" * 500 + b"C" * 700
    assert accepted == 2
    assert received_bytes == wire.sent_bytes - len(wire.packets[1])


def test_one_erasure_parity_recovers_exact_layer_and_counts_budget():
    class Collector:
        def decode_bytes(self, payload: bytes) -> bytes:
            return payload

    class StaticCodec:
        def encode_full(self, image: Image.Image, *, focus_pixels: np.ndarray | None):
            return stream

    stream = LimitNetStream(b"header", (b"A" * 1500, b"B" * 700))
    legacy = _limit_wire(stream, 2)
    protected = _limit_wire(stream, 2, parity_packets_per_layer=1)
    assert len(legacy.packets) == 3
    assert len(protected.packets) == 5  # 每层独立增加一个校验包。
    assert protected.sent_bytes == legacy.sent_bytes + len(protected.packets[2]) + len(protected.packets[4])

    for lost_index in (0, 1):
        lost = tuple(index == lost_index for index in range(len(protected.packets)))
        decoded, accepted, received_bytes = _receive_limit(protected, lost, Collector())
        assert decoded == stream.to_bytes()
        assert accepted == 2
        assert received_bytes == protected.sent_bytes - len(protected.packets[lost_index])

    # 校验包丢失时两个数据层仍完整；同层丢两个数据包则只保留后续层。
    decoded, accepted, _ = _receive_limit(
        protected, (False, False, True, False, False), Collector())
    assert decoded == stream.to_bytes() and accepted == 2
    decoded, accepted, _ = _receive_limit(
        protected, (True, True, False, False, False), Collector())
    assert decoded == b"header" + stream.layers[1] and accepted == 1

    one_layer = _limit_wire(stream, 1, parity_packets_per_layer=1)
    wire, count = _limit_under_budget(StaticCodec(), _sample_image(), None,
                                      one_layer.sent_bytes,
                                      parity_packets_per_layer=1)
    assert count == 1 and wire.sent_bytes == one_layer.sent_bytes
