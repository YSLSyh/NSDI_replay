"""单码流 JPEG 预算搜索只依赖输入图像和发送端重要性图。"""

from io import BytesIO
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replay.transmission.task_jpeg_search import (  # noqa: E402
    SearchProfile, _distortion, search_task_jpeg,
)
import replay.transmission.task_jpeg_search as search_module  # noqa: E402
from replay.transmission.plain_jpeg_channel import plain_jpeg_wire_bytes  # noqa: E402
from replay.transmission.jpeg import (  # noqa: E402
    _prequantize, _restore_protected_rgb, encode_task_jpeg,
)


def _fixture() -> tuple[Image.Image, np.ndarray]:
    y, x = np.mgrid[:128, :128]
    array = np.stack(((3 * x + y) % 256, (x + 5 * y) % 256,
                      (7 * x + 2 * y) % 256), axis=-1).astype(np.uint8)
    importance = np.full((8, 8), 0.01, dtype=np.float32)
    importance[2:5, 2:6] = 1.0
    return Image.fromarray(array, "RGB"), importance


class TaskJpegSearchTests(unittest.TestCase):
    def test_single_standard_jpeg_with_exact_packet_budget_and_no_restart(self):
        image, importance = _fixture()
        payload, report = search_task_jpeg(
            image, importance, budget_bytes=12000,
            profiles=(SearchProfile(0), SearchProfile(1)),
            selected_patch_mask=importance > 0.5)
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        self.assertTrue(payload.endswith(b"\xff\xd9"))
        self.assertNotIn(b"\xff\xdd", payload)  # DRI marker
        for marker in range(0xD0, 0xD8):
            self.assertNotIn(bytes((0xFF, marker)), payload)
        with Image.open(BytesIO(payload)) as decoded:
            self.assertEqual(decoded.size, image.size)
        expected_wire = plain_jpeg_wire_bytes(payload)
        self.assertEqual(report["codec_wire_bytes"], expected_wire)
        self.assertLessEqual(expected_wire, 12000)
        self.assertEqual(report["semantic_mask_source"], "selected_patch_mask")
        self.assertEqual(report["selected_patch_count"], 12)
        self.assertFalse(report["selection_uses_validation_labels"])
        self.assertEqual(report["restart_marker_rows"], 0)

    def test_candidate_selection_uses_lowest_measured_weighted_distortion(self):
        image, importance = _fixture()
        _, report = search_task_jpeg(
            image, importance, budget_bytes=12000,
            profiles=(SearchProfile(0), SearchProfile(1), SearchProfile(2)))
        scores = [item["weighted_rgb_mse"] for item in report["candidate_metrics"]]
        self.assertAlmostEqual(report["weighted_rgb_mse"], min(scores))
        self.assertGreater(report["evaluated_feasible_candidates"], 1)

    def test_plain_jpeg_fallback_and_infeasible_budget(self):
        image, importance = _fixture()
        payload, report = search_task_jpeg(
            image, importance, budget_bytes=12000,
            profiles=(SearchProfile(0),))
        self.assertTrue(report["plain_jpeg"])
        self.assertGreater(len(payload), 0)
        with self.assertRaisesRegex(ValueError, "无法满足"):
            search_task_jpeg(image, importance, budget_bytes=100,
                             profiles=(SearchProfile(0),))
        with self.assertRaisesRegex(ValueError, "兜底候选"):
            search_task_jpeg(image, importance, budget_bytes=12000,
                             profiles=(SearchProfile(2),))

    def test_frozen_task_profile_never_selects_plain_jpeg(self):
        image, importance = _fixture()
        profile = SearchProfile(2.0, 0.0, True, 2,
                                background_low_frequency_strength=12.0)
        _, report = search_task_jpeg(
            image, importance, budget_bytes=12000,
            selected_patch_mask=importance > 0.5,
            profiles=(profile,), require_plain_fallback=False)
        self.assertFalse(report["plain_jpeg"])
        self.assertFalse(report["require_plain_fallback"])
        self.assertEqual(report["candidate_profiles"], 1)

    def test_uniform_zero_importance_has_no_roi_and_still_returns_jpeg(self):
        image, importance = _fixture()
        payload, report = search_task_jpeg(
            image, np.zeros_like(importance), budget_bytes=12000)
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        self.assertEqual(report["selected_patch_count"], 0)
        self.assertIsNone(report["roi_rgb_mse"])
        self.assertAlmostEqual(report["weighted_rgb_mse"],
                               report["background_rgb_mse"])

    def test_selected_mask_shape_is_checked(self):
        image, importance = _fixture()
        with self.assertRaisesRegex(ValueError, "布尔 patch 网格"):
            search_task_jpeg(image, importance, budget_bytes=12000,
                             selected_patch_mask=np.zeros((2, 2), dtype=bool))

    def test_boolean_ground_truth_importance_reaches_jpeg_encoder_unchanged(self):
        image, _ = _fixture()
        gt_mask = np.zeros((8, 8), dtype=bool)
        gt_mask[2:4, 3:5] = True
        seen_dtypes = []
        original_encode = search_module.encode_task_jpeg

        def record_importance(*args, **kwargs):
            seen_dtypes.append(np.asarray(args[1]).dtype)
            return original_encode(*args, **kwargs)

        with patch.object(search_module, "encode_task_jpeg",
                          side_effect=record_importance):
            payload, report = search_task_jpeg(
                image, gt_mask, budget_bytes=12000,
                selected_patch_mask=gt_mask, max_quality=15,
                profiles=(SearchProfile(0.0), SearchProfile(2.0, 0.0, True, 2)))
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        self.assertTrue(seen_dtypes)
        self.assertTrue(all(dtype == np.dtype(bool) for dtype in seen_dtypes))
        self.assertEqual(report["selected_patch_count"], 4)
        self.assertEqual(report["semantic_mask_source"], "selected_patch_mask")

    def test_tiny_roi_is_weighted_by_region_not_pixel_count(self):
        source = np.zeros((10, 10, 3), dtype=np.float32)
        restored = np.ones((10, 10, 3), dtype=np.uint8)
        restored[0, 0] = 10
        selected = np.zeros((10, 10), dtype=bool)
        selected[0, 0] = True
        score, roi_mse, bg_mse = _distortion(
            source, Image.fromarray(restored), selected, 4.0, 1.0)
        self.assertEqual(roi_mse, 100.0)
        self.assertEqual(bg_mse, 1.0)
        self.assertAlmostEqual(score, 80.2)

    def test_selected_patch_is_not_prequantized(self):
        image, importance = _fixture()
        protected = np.zeros((8, 8), dtype=bool)
        protected[0, 0] = True
        prepared = _prequantize(
            image, importance, 16, 45, 0, 2, 6.0, 6.0, 1.0,
            1.0, 4.0, fast_prequant=True, protected_patch_mask=protected)
        source_ycbcr = np.asarray(image.convert("YCbCr"))
        actual_ycbcr = np.asarray(prepared)
        self.assertTrue(np.array_equal(actual_ycbcr[:16, :16],
                                       source_ycbcr[:16, :16]))
        self.assertFalse(np.array_equal(actual_ycbcr[16:32, 16:32],
                                        source_ycbcr[16:32, 16:32]))
        # 送入单一标准 JPEG 前，以 RGB 原像素覆盖被选区域。
        final = np.asarray(_restore_protected_rgb(prepared, image, protected, 16))
        original = np.asarray(image)
        self.assertTrue(np.array_equal(final[:16, :16], original[:16, :16]))

    def test_opt_in_background_low_frequency_quantization_preserves_roi(self):
        pixels = np.random.default_rng(14).integers(
            0, 256, size=(128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, "RGB")
        selected = np.zeros((8, 8), dtype=bool)
        selected[2:4, 3:5] = True
        prepared = _prequantize(
            image, selected, 16, 35, 0, 2, 0.0, 0.0, 1.0,
            1.0, 4.0, fast_prequant=True,
            protected_patch_mask=selected,
            background_low_frequency_strength=48.0)
        original_ycbcr = np.asarray(image.convert("YCbCr"))
        actual_ycbcr = np.asarray(prepared)
        np.testing.assert_array_equal(actual_ycbcr[32:64, 48:80],
                                      original_ycbcr[32:64, 48:80])
        self.assertTrue(np.any(actual_ycbcr[:16, :16] != original_ycbcr[:16, :16]))

        # 没有目标的样本也应把整图当背景；旧参数路径保持原有输出。
        empty = np.zeros((8, 8), dtype=bool)
        background = _prequantize(
            image, empty, 16, 35, 0, 2, 0.0, 0.0, 1.0,
            1.0, 4.0, fast_prequant=True,
            background_low_frequency_strength=48.0)
        self.assertTrue(np.any(np.asarray(background) != original_ycbcr))
        unchanged = _prequantize(
            image, empty, 16, 35, 0, 2, 0.0, 0.0, 1.0, 1.0, 4.0)
        self.assertIs(unchanged, image)

    def test_opt_in_low_frequency_reduces_bytes_and_keeps_standard_jpeg(self):
        pixels = np.random.default_rng(29).integers(
            0, 256, size=(128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, "RGB")
        selected = np.zeros((8, 8), dtype=bool)
        selected[2:4, 3:5] = True
        kwargs = dict(luma_strength=32.0, chroma_strength=32.0,
                      zero_threshold_multiplier=1.0, fast_prequant=True,
                      protected_patch_mask=selected, optimize_huffman=True)
        old_payload = encode_task_jpeg(
            image, selected, 16, 35, 0, 2, **kwargs)
        new_payload = encode_task_jpeg(
            image, selected, 16, 35, 0, 2,
            background_low_frequency_strength=48.0, **kwargs)
        self.assertLess(len(new_payload), len(old_payload))
        self.assertTrue(new_payload.startswith(b"\xff\xd8"))
        self.assertTrue(new_payload.endswith(b"\xff\xd9"))
        self.assertNotIn(b"\xff\xdd", new_payload)
        with Image.open(BytesIO(new_payload)) as decoded:
            decoded.load()
            self.assertEqual(decoded.size, image.size)

    def test_low_frequency_profile_is_reported_for_train_only_tuning(self):
        image, _ = _fixture()
        selected = np.zeros((8, 8), dtype=bool)
        selected[2:4, 3:5] = True
        _, report = search_task_jpeg(
            image, selected, budget_bytes=12_000, selected_patch_mask=selected,
            max_quality=20,
            profiles=(SearchProfile(0.0),
                      SearchProfile(8.0, 0.0, True, 2,
                                    background_low_frequency_strength=12.0)))
        values = {record["background_low_frequency_strength"]
                  for record in report["candidate_metrics"]}
        self.assertIn(0.0, values)
        self.assertIn(12.0, values)

