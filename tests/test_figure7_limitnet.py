"""图 7 LimitNet 的预算、官方权重来源和分层丢包适配。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from replay.baselines.limitnet import LimitNetStream
from replay.evaluation.figure7_limitnet import (
    Figure7LimitNetSpec, LimitNetClassificationCodec,
    verify_official_imagenet_limitnet,
)


class FakeLimitCodec:
    source = "official_limitnet_imagenet_pretrained"

    def encode_full(self, image, *, focus_pixels=None):
        if image.size != (224, 224) or focus_pixels is not None:
            raise AssertionError("图 7 LimitNet 不应读取真值关注区域")
        return LimitNetStream(b"fake-header", (b"a" * 900, b"b" * 1100))

    def decode_bytes(self, payload):
        if not payload.startswith(b"fake-header"):
            raise ValueError("头损坏")
        return Image.new("RGB", (224, 224), "gray")


class Figure7LimitNetTests(unittest.TestCase):
    _TEMP_PARENT = Path(__file__).resolve().parents[1] / "artifacts"

    def test_codec_counts_exact_wire_bytes_and_resizes_for_shared_classifier(self) -> None:
        codec = LimitNetClassificationCodec(
            FakeLimitCodec(), fixed_loss_budget_bytes=3000,
            provenance={"source": "official_limitnet_imagenet_pretrained"})
        original = Image.new("RGB", (300, 180), "red")
        payload, meta = codec.encode(original, budget_bytes=1200)
        self.assertEqual(len(payload), meta["wire_bytes"])
        self.assertLessEqual(meta["wire_bytes"], 1200)
        self.assertGreater(meta["layers_sent"], 0)
        self.assertEqual(meta["loss_packet_count"], len(payload.wire.packets))
        decoded = codec.decode(payload, loss_flags=(False,) * len(payload.wire.packets),
                               received_prefix_bytes=None)
        self.assertEqual(decoded.size, original.size)

    def test_official_raw_md5_is_required(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            root = Path(temp)
            raw = root / "LimitNet-ImageNet"
            converted = root / "converted.pt"
            raw.write_bytes(b"not official")
            converted.write_bytes(b"not a checkpoint")
            spec = Figure7LimitNetSpec(raw, converted, fixed_loss_budget_bytes=30_000)
            with self.assertRaisesRegex(ValueError, "MD5"):
                verify_official_imagenet_limitnet(spec)

    def test_infeasible_budget_has_no_fake_decoded_image(self) -> None:
        codec = LimitNetClassificationCodec(
            FakeLimitCodec(), fixed_loss_budget_bytes=3000,
            provenance={"source": "official_limitnet_imagenet_pretrained"})
        payload, metadata = codec.encode(Image.new("RGB", (224, 224)), budget_bytes=1)
        self.assertEqual(metadata["wire_bytes"], 0)
        self.assertIsNone(codec.decode(payload, loss_flags=(), received_prefix_bytes=None))


if __name__ == "__main__":
    unittest.main()
