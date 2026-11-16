"""普通与渐进 JPEG 基线必须输出真实码流、保持原尺寸和字节预算。"""

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from replay.transmission.baselines import (
    encode_baseline_fixed_quality, encode_baseline_under_budget,
)


class JpegBaselineTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(
            np.random.default_rng(15).integers(0, 256, (39, 57, 3), dtype=np.uint8))

    def test_progressive_and_standard_are_distinct_decodable_jpeg_streams(self):
        for progressive in (False, True):
            with self.subTest(progressive=progressive):
                method = "progressive_jpeg" if progressive else "jpeg"
                payload = encode_baseline_fixed_quality(self.image, method, 75, 0)
                self.assertTrue(payload.startswith(b"\xff\xd8"))
                with Image.open(BytesIO(payload)) as decoded:
                    self.assertEqual(decoded.size, self.image.size)
                    self.assertEqual(bool(decoded.info.get("progressive")), progressive)

    def test_budget_search_charges_real_stream_bytes(self):
        for progressive in (False, True):
            with self.subTest(progressive=progressive):
                method = "progressive_jpeg" if progressive else "jpeg"
                payload, info = encode_baseline_under_budget(self.image, method, 1800, 0)
                self.assertLessEqual(len(payload), 1800)
                self.assertEqual(info["jpeg_bytes"], len(payload))
                self.assertGreaterEqual(info["quality"], 1)
                self.assertEqual(info["method"], method)
                self.assertEqual(info["original_size"], [57, 39])
                with Image.open(BytesIO(payload)) as decoded:
                    self.assertEqual(decoded.size, self.image.size)

    def test_impossible_budget_does_not_resize(self):
        with self.assertRaisesRegex(ValueError, "原尺寸下无法满足"):
            encode_baseline_under_budget(self.image, "jpeg", 100, 0)

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持的图像基线"):
            encode_baseline_fixed_quality(self.image, "fake")


if __name__ == "__main__":
    unittest.main()
