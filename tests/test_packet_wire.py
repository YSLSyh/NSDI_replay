"""验证应用层 JPEG 分包线格式与丢失后的接收端重组。"""

from io import BytesIO
import math
import unittest

import numpy as np
from PIL import Image

from replay.transmission.packet_loss import plan_entropy_packets, simulate_restart_stream_loss
from replay.transmission.packet_wire import (
    CONTROL_FIXED_BYTES,
    INTERVAL_MANIFEST_BYTES,
    pack_packetized_frame,
    receive_packetized_frame,
)


def _frame():
    pixels = np.random.default_rng(8).integers(
        0, 256, size=(64, 128, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=90, subsampling=0,
               restart_marker_blocks=math.ceil(image.width / 16))
    payload = stream.getvalue()
    _, _, intervals, packets = plan_entropy_packets(payload, 300)
    frame = pack_packetized_frame(b"session", b"frame", payload, intervals, packets,
                                  packet_payload_bytes=300, packet_header_bytes=8)
    return payload, intervals, packets, frame


class PacketWireTests(unittest.TestCase):
    def test_lossless_wire_round_trip_and_exact_byte_accounting(self):
        payload, intervals, packets, frame = _frame()
        received = receive_packetized_frame(frame.control, frame.packets)
        self.assertEqual(received.session_header, b"session")
        self.assertEqual(received.frame_header, b"frame")
        self.assertEqual(received.jpeg_payload, payload)
        self.assertEqual(received.lost_packet_indices, ())
        self.assertEqual(received.lost_intervals, ())
        self.assertEqual(received.lost_entropy_bytes, 0)
        self.assertEqual(received.receiver_discarded_bytes, 0)
        expected = (CONTROL_FIXED_BYTES + len(b"session") + len(b"frame") +
                    len(payload) + INTERVAL_MANIFEST_BYTES * len(intervals) +
                    8 * len(packets))
        self.assertEqual(frame.sent_bytes, expected)
        self.assertEqual(received.received_bytes, expected)
        with Image.open(BytesIO(received.jpeg_payload)) as image:
            image.load()
            self.assertEqual(image.size, (128, 64))

    def test_lost_or_corrupted_fragment_discards_its_interval(self):
        payload, intervals, packets, frame = _frame()
        # 选多片区间：物理丢一个包，接收端丢该区间其余已收到熵字节。
        dropped_index = next(index for index, packet in enumerate(packets)
                             if sum(other.interval_ordinal == packet.interval_ordinal
                                    for other in packets) > 1)
        interval = intervals[packets[dropped_index].interval_ordinal]
        delivered = tuple(packet for index, packet in enumerate(frame.packets)
                          if index != dropped_index)
        received = receive_packetized_frame(frame.control, delivered)
        expected_lost_bytes = packets[dropped_index].end - packets[dropped_index].start
        self.assertEqual(received.lost_packet_indices, (dropped_index,))
        self.assertEqual(received.lost_intervals, ((interval.scan, interval.index),))
        self.assertEqual(received.lost_entropy_bytes, expected_lost_bytes)
        self.assertEqual(received.receiver_discarded_bytes,
                         interval.end - interval.start - expected_lost_bytes)
        self.assertEqual(len(payload) - len(received.jpeg_payload),
                         interval.end - interval.start)
        self.assertEqual(received.received_bytes,
                         frame.sent_bytes - len(frame.packets[dropped_index]))

        # 坏包虽占用链路字节，CRC 错误后必须等效为缺包。
        broken = bytearray(frame.packets[dropped_index])
        broken[-1] ^= 0x01
        wire_packets = list(frame.packets)
        wire_packets[dropped_index] = bytes(broken)
        corrupted = receive_packetized_frame(frame.control, wire_packets)
        self.assertEqual(corrupted.lost_packet_indices, received.lost_packet_indices)
        self.assertEqual(corrupted.jpeg_payload, received.jpeg_payload)
        self.assertEqual(corrupted.received_bytes, frame.sent_bytes)

        trace = [index == dropped_index for index in range(len(packets))]
        simulated = simulate_restart_stream_loss(
            payload, image_id="wire-test", scenario="scooter", level="medium",
            seed=1, packet_payload_bytes=300, loss_trace=trace)
        self.assertEqual(simulated.received_payload, received.jpeg_payload)
        self.assertEqual(simulated.receiver_discarded_bytes,
                         received.receiver_discarded_bytes)

    def test_corrupted_control_manifest_is_rejected(self):
        _, _, _, frame = _frame()
        broken = bytearray(frame.control)
        broken[len(broken) // 2] ^= 0x01
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            receive_packetized_frame(bytes(broken), frame.packets)

    def test_corrupted_packet_index_cannot_misplace_valid_payload(self):
        _, _, packets, frame = _frame()
        source = next(index for index in range(len(packets))
                      if (index ^ 1) < len(packets) and
                      len(frame.packets[index]) == len(frame.packets[index ^ 1]))
        target = source ^ 1
        # 仅翻转包序号中的 1 bit，指向另一个尚未送达的同长包。
        forged = bytearray(frame.packets[source])
        forged[:4] = target.to_bytes(4, "big")
        delivered = [packet for index, packet in enumerate(frame.packets)
                     if index not in (source, target)]
        delivered.append(bytes(forged))
        received = receive_packetized_frame(frame.control, delivered)
        self.assertIn(source, received.lost_packet_indices)
        self.assertIn(target, received.lost_packet_indices)


if __name__ == "__main__":
    unittest.main()
