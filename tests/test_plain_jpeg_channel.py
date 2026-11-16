"""无 RST JPEG 保留结构、真实丢失熵分片的核验。"""

from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from replay.transmission.baselines import encode_jpeg_baseline
from replay.transmission.plain_jpeg_channel import (
    plain_jpeg_wire_bytes, simulate_plain_jpeg_entropy_loss, split_plain_jpeg_scan,
)


class PlainJpegChannelTests(unittest.TestCase):
    def test_reliable_markers_counted_and_entropy_loss_real(self) -> None:
        yy, xx = np.indices((128, 128))
        rgb = np.stack((xx * 2, yy * 2, (xx + yy) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(rgb)
        payload = encode_jpeg_baseline(image, 75, 0, optimize_huffman=True)
        prefix, entropy, suffix = split_plain_jpeg_scan(payload)
        self.assertEqual(prefix + entropy + suffix, payload)
        self.assertEqual(suffix, b"\xff\xd9")
        packet_count = (len(entropy) + 1199) // 1200
        intact = simulate_plain_jpeg_entropy_loss(
            payload, image_id="scene", expected_size=image.size,
            loss_trace=(False,) * packet_count)
        self.assertEqual(intact.received_payload, payload)
        self.assertEqual(intact.status, "ok")
        self.assertEqual(intact.reliable_control_bytes, len(prefix) + len(suffix))
        self.assertEqual(intact.sent_bytes, plain_jpeg_wire_bytes(payload))
        damaged = simulate_plain_jpeg_entropy_loss(
            payload, image_id="scene", expected_size=image.size,
            loss_trace=(True,) + (False,) * (packet_count - 1))
        self.assertEqual(damaged.lost_packet_indices, (0,))
        self.assertEqual(damaged.received_payload[:len(prefix)], prefix)
        self.assertEqual(damaged.received_payload[-len(suffix):], suffix)
        self.assertEqual(damaged.received_bytes, damaged.sent_bytes -
                         damaged.lost_entropy_bytes - 8)
        # JPEG 库自己判定受损码流可否解码，测试不人为规定结果。
        if damaged.decoded is not None:
            with Image.open(BytesIO(damaged.received_payload)) as candidate:
                self.assertEqual(candidate.size, image.size)
        else:
            self.assertIsNotNone(damaged.decode_error)


if __name__ == "__main__":
    unittest.main()
