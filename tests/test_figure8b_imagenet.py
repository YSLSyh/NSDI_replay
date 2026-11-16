"""图 8(b) 新分类评测的配对丢包与分母核验。"""

from pathlib import Path
import unittest

from PIL import Image

from replay.evaluation.figure7_classification import ValSample
from replay.evaluation.figure8b_imagenet import (
    evaluate_image_resolution, pilot_samples, psnr_rgb,
    resize_center_crop, summarize_rows,
)


class FakeClassifier:
    def predict(self, image: Image.Image) -> int:
        return 0 if image.getpixel((0, 0))[0] > 100 else 1


class FakeCodec:
    def __init__(self, name: str, *, fails: bool = False):
        self.name = name
        self.fails = fails
        self.loss_flags = None
        self.size = None

    def encode(self, image: Image.Image, *, budget_bytes: int | None):
        self.size = image.size
        return b"x" * 2400, {"wire_bytes": 2400, "loss_packet_count": 2}

    def decode(self, payload: bytes, *, loss_flags: tuple[bool, ...],
               received_prefix_bytes: int | None):
        self.loss_flags = loss_flags
        if self.fails:
            return None
        color = (200, 10, 10) if self.name == "kdc" else (20, 10, 10)
        return Image.new("RGB", self.size, color)


class Figure8bImageNetTests(unittest.TestCase):
    def test_resize_center_crop_preserves_square_without_stretching(self) -> None:
        source = Image.new("RGB", (120, 60), (170, 20, 20))
        cropped = resize_center_crop(source, 64)
        self.assertEqual(cropped.size, (64, 64))
        self.assertEqual(cropped.getpixel((32, 32))[0], 170)
        with self.assertRaises(ValueError):
            resize_center_crop(source, 0)
        with self.assertRaises(ValueError):
            psnr_rgb(source, cropped)

    def test_paired_loss_and_failed_decoding_counted_as_error(self) -> None:
        sample = ValSample("n00000001/ILSVRC2012_val_00000001.JPEG",
                           Path("unused.jpeg"), 0, "n00000001")
        source = Image.new("RGB", (80, 60), (180, 10, 10))
        limitnet = FakeCodec("limitnet", fails=True)
        kdc = FakeCodec("kdc")
        rows = evaluate_image_resolution(
            sample=sample, image=source, resolution=64,
            classifier=FakeClassifier(), codecs={"kdc": kdc, "limitnet": limitnet},
            budget_bytes=3000, loss_rate=0.5, channel_seed=7)
        self.assertEqual(limitnet.loss_flags, kdc.loss_flags)
        self.assertEqual(rows[0]["channel_event_id"], rows[1]["channel_event_id"])
        self.assertEqual(rows[0]["status"], "decode_failed")
        self.assertEqual(rows[0]["prediction"], -1)
        self.assertEqual(rows[1]["status"], "ok")
        summary = summarize_rows(rows, [sample], (64,))
        self.assertEqual(summary[0]["top1"], 0)
        self.assertIsNone(summary[0]["psnr_db"])
        self.assertEqual(summary[1]["top1"], 1)
        self.assertEqual(summary[1]["psnr_denominator"], 1)
        self.assertEqual(summary[1]["intact_top1"], 1)
        with self.assertRaisesRegex(ValueError, "不完整"):
            summarize_rows(rows[:1], [sample], (64,))

    def test_pilot_requires_explicit_strict_subset(self) -> None:
        samples = [ValSample(str(i), Path("unused"), 0, "n00000001")
                   for i in range(10)]
        self.assertIs(pilot_samples(samples, count=None, seed=7), samples)
        self.assertEqual(len(pilot_samples(samples, count=3, seed=7)), 3)
        with self.assertRaises(ValueError):
            pilot_samples(samples, count=10, seed=7)
