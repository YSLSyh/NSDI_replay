"""验证应用层分片丢失、JPEG 重同步及接收端字节核算。"""

from io import BytesIO
import math
import unittest

import numpy as np
from PIL import Image

from replay.transmission.packet_loss import (
    count_entropy_intervals,
    count_entropy_packets,
    plan_entropy_packets,
    simulate_restart_stream_loss,
)


def _image() -> Image.Image:
    pixels = np.random.default_rng(4).integers(
        0, 256, size=(65, 136, 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


def _jpeg(image: Image.Image, *, progressive: bool = False) -> bytes:
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=87, subsampling=0,
               progressive=progressive,
               restart_marker_blocks=math.ceil(image.width / 16))
    return stream.getvalue()


class PacketLossTests(unittest.TestCase):
    def test_actual_dri_and_packet_plan_across_scans(self):
        image = _image()
        for progressive in (False, True):
            payload = _jpeg(image, progressive=progressive)
            width, height, intervals, packets = plan_entropy_packets(payload, 500)
            self.assertEqual((width, height), image.size)
            self.assertEqual(len(intervals), count_entropy_intervals(payload))
            self.assertGreaterEqual(len(packets), count_entropy_packets(payload))
            self.assertTrue(all(0 < packet.end - packet.start <= 500
                                for packet in packets))
            self.assertEqual([packet.interval_ordinal for packet in packets],
                             sorted(packet.interval_ordinal for packet in packets))
            self.assertTrue(all(packet.scan == intervals[packet.interval_ordinal].scan and
                                packet.interval == intervals[packet.interval_ordinal].index
                                for packet in packets))
            total_mcus = math.ceil(width / 8) * math.ceil(height / 8)
            for scan in sorted({interval.scan for interval in intervals}):
                scan_intervals = [interval for interval in intervals if interval.scan == scan]
                self.assertEqual(scan_intervals[0].mcu_start, 0)
                self.assertEqual(scan_intervals[-1].mcu_end, total_mcus)
                self.assertTrue(all(first.mcu_end == second.mcu_start
                                    for first, second in zip(scan_intervals, scan_intervals[1:])))

    def test_one_lost_fragment_discards_only_its_restart_interval(self):
        payload = _jpeg(_image())
        width, height, intervals, packets = plan_entropy_packets(payload, 300)
        # 找到跨多个分片的区间，仅让其中一个分片在网络中丢失。
        chosen = next(index for index, packet in enumerate(packets)
                      if sum(other.interval_ordinal == packet.interval_ordinal
                             for other in packets) > 1)
        trace = [False] * len(packets)
        trace[chosen] = True
        result = simulate_restart_stream_loss(
            payload, image_id="frame", scenario="scooter", level="medium", seed=4,
            packet_payload_bytes=300, packet_header_bytes=8, loss_trace=trace)
        interval = intervals[packets[chosen].interval_ordinal]
        lost_length = packets[chosen].end - packets[chosen].start
        self.assertEqual(result.loss_model, "external_trace")
        self.assertEqual(result.lost_packet_indices, (chosen,))
        self.assertEqual(result.lost_intervals, ((interval.scan, interval.index),))
        self.assertEqual(result.lost_entropy_bytes, lost_length)
        self.assertEqual(result.receiver_discarded_bytes,
                         interval.end - interval.start - lost_length)
        self.assertEqual(result.sent_bytes - result.received_bytes, lost_length + 8)
        self.assertEqual(len(payload) - len(result.received_payload),
                         interval.end - interval.start)
        self.assertEqual(result.status, "ok", result.decode_error)

        # DRI 控制一个 MCU 序列，区间可能跨行；掩码必须按实际 MCU 映射。
        expected = np.zeros((height, width), dtype=np.float32)
        mcus_per_row = math.ceil(width / 8)
        for mcu in range(interval.mcu_start, interval.mcu_end):
            row, col = divmod(mcu, mcus_per_row)
            expected[row * 8:min((row + 1) * 8, height),
                     col * 8:min((col + 1) * 8, width)] = 1
        np.testing.assert_array_equal(result.missing_pixels, expected)
        self.assertLess(float(expected.mean()), 0.5)  # 不能默认抹掉整张或整行。

    def test_synthetic_events_are_reproducible_and_common_by_packet_index(self):
        image = _image()
        baseline = _jpeg(image)
        progressive = _jpeg(image, progressive=True)
        for loss_model in ("bernoulli", "gilbert_elliott"):
            outcomes = [simulate_restart_stream_loss(
                payload, image_id="same-frame", scenario="scooter", level="high",
                seed=72, packet_payload_bytes=300, loss_model=loss_model)
                for payload in (baseline, baseline, progressive)]
            self.assertEqual(outcomes[0].lost_packet_indices, outcomes[1].lost_packet_indices)
            count = min(outcomes[0].total_packets, outcomes[2].total_packets)
            first = {index for index in outcomes[0].lost_packet_indices if index < count}
            other = {index for index in outcomes[2].lost_packet_indices if index < count}
            self.assertEqual(first, other)
            self.assertEqual(outcomes[0].loss_model, loss_model)

    def test_external_trace_requires_all_packet_events(self):
        payload = _jpeg(_image())
        with self.assertRaisesRegex(ValueError, "trace"):
            simulate_restart_stream_loss(
                payload, image_id="frame", scenario="scooter", level="medium",
                seed=4, loss_trace=[False])

    def test_rejects_jpeg_without_restart_markers(self):
        image = _image()
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=60, subsampling=0)
        with self.assertRaisesRegex(ValueError, "DRI"):
            count_entropy_packets(stream.getvalue())


if __name__ == "__main__":
    unittest.main()
