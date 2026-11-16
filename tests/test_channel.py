"""信道代理的公平性与可重放性测试。"""

import unittest

import numpy as np
from PIL import Image

from replay.transmission.channel import simulate_token_erasure


class ChannelTests(unittest.TestCase):
    def test_reproducible_and_shared_across_codecs(self):
        first = Image.new("RGB", (160, 96), (180, 30, 20))
        second = Image.new("RGB", (160, 96), (20, 140, 200))
        args = {"image_id": "city/frame_1", "scenario": "scooter",
                "level": "medium", "seed": 2026}
        a = simulate_token_erasure(first, transmitted_bytes=14000, **args)
        b = simulate_token_erasure(second, transmitted_bytes=9000, **args)
        c = simulate_token_erasure(first, transmitted_bytes=14000, **args)
        np.testing.assert_array_equal(a.lost_patches, b.lost_patches)
        np.testing.assert_array_equal(a.missing_pixels, b.missing_pixels)
        np.testing.assert_array_equal(a.missing_pixels, c.missing_pixels)
        self.assertEqual(a.throughput_mbps, b.throughput_mbps)
        self.assertGreater(a.transfer_latency_ms, b.transfer_latency_ms)
        self.assertEqual(a.transfer_latency_ms, c.transfer_latency_ms)
        self.assertEqual(a.proxy_kind, "post_jpeg_decode_semantic_token_erasure")

    def test_native_size_and_partial_edge_patches(self):
        image = Image.new("RGB", (57, 39), (200, 100, 50))
        result = simulate_token_erasure(image, image_id="edge", scenario="scooter",
                                        level="high", seed=4, transmitted_bytes=500)
        self.assertEqual(result.damaged.size, image.size)
        self.assertEqual(result.lost_patches.shape, (3, 4))
        self.assertEqual(result.missing_pixels.shape, (39, 57))
        expected = np.repeat(np.repeat(result.lost_patches, 16, axis=0),
                             16, axis=1)[:39, :57]
        np.testing.assert_array_equal(result.missing_pixels, expected.astype(np.float32))
        damaged = np.asarray(result.damaged)
        self.assertTrue(np.all(damaged[expected] == 0))
        self.assertTrue(np.all(damaged[~expected] == [200, 100, 50]))

    def test_loss_levels_are_nested_and_scenario_ranges_hold(self):
        image = Image.new("RGB", (640, 320), (100, 100, 100))
        for scenario, range_mbps, target_rates in (
            ("highway", (5, 35), (0.02, 0.05, 0.08)),
            ("scooter", (10, 31), (0.03, 0.075, 0.12)),
        ):
            results = [simulate_token_erasure(
                image, image_id="same-frame", scenario=scenario, level=level,
                seed=11, transmitted_bytes=25000,
            ) for level in ("low", "medium", "high")]
            self.assertEqual([item.target_loss_rate for item in results], list(target_rates))
            self.assertLessEqual(results[0].lost_patch_count, results[1].lost_patch_count)
            self.assertLessEqual(results[1].lost_patch_count, results[2].lost_patch_count)
            self.assertTrue(np.all(results[0].lost_patches <= results[1].lost_patches))
            self.assertTrue(np.all(results[1].lost_patches <= results[2].lost_patches))
            for result in results:
                self.assertGreaterEqual(result.throughput_mbps, range_mbps[0])
                self.assertLessEqual(result.throughput_mbps, range_mbps[1])
                self.assertAlmostEqual(result.actual_loss_rate,
                                       result.lost_patch_count / result.total_patch_count)

    def test_invalid_inputs_rejected(self):
        image = Image.new("RGB", (32, 32))
        params = {"image_id": "a", "scenario": "highway", "level": "low",
                  "seed": 0, "transmitted_bytes": 1}
        for change in ({"scenario": "unknown"}, {"level": "unknown"},
                       {"transmitted_bytes": -1}, {"patch_size": 0},
                       {"fixed_overhead_ms": float("nan")}, {"image_id": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                simulate_token_erasure(image, **(params | change))


if __name__ == "__main__":
    unittest.main()
