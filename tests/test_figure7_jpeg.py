"""图 7 JPEG 对照：普通 JPEG 完整质量搜索、渐进扫描和真实传输字节数。"""

from __future__ import annotations

from math import ceil
import unittest

from PIL import Image

from replay.baselines.figure7_jpeg import (
    _scan_end_offsets,
    JpegClassificationCodec,
    decode_received_jpeg,
    deliver_jpeg_prefix,
    encode_jpeg,
    jpeg_wire_bytes,
    search_jpeg_budget,
)


class Figure7JpegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        image = Image.new("RGB", (128, 96))
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                pixels[x, y] = ((x * 13 + y * 5) % 256,
                                (x * 3 + y * 11) % 256,
                                (x + y * 2) % 256)
        cls.image = image

    def test_fixed_quality_streams_preserve_resolution(self) -> None:
        for progressive in (False, True):
            with self.subTest(progressive=progressive):
                stream = encode_jpeg(self.image, progressive=progressive)
                flags = (False,) * ceil(len(stream) / 1200)
                decoded = decode_received_jpeg(
                    stream, loss_flags=flags, progressive=progressive)
                self.assertIsNotNone(decoded)
                self.assertEqual(decoded.size, self.image.size)
        self.assertGreater(len(_scan_end_offsets(
            encode_jpeg(self.image, progressive=True))), 1)

    def test_budget_progressive_decodes_complete_scan_only(self) -> None:
        stream = encode_jpeg(self.image, progressive=True)
        first_scan = _scan_end_offsets(stream)[0]
        insufficient = deliver_jpeg_prefix(
            stream, jpeg_wire_bytes(first_scan - 1, packet_header_bytes=8),
            progressive=True, packet_header_bytes=8)
        self.assertIsNone(insufficient.image)
        self.assertEqual(insufficient.decoded_codec_bytes, 0)
        budget = jpeg_wire_bytes(first_scan + 50, packet_header_bytes=8)
        outcome = deliver_jpeg_prefix(
            stream, budget, progressive=True, packet_header_bytes=8)
        self.assertIsNotNone(outcome.image)
        self.assertEqual(outcome.image.size, self.image.size)
        self.assertEqual(outcome.decoded_codec_bytes, first_scan)
        self.assertEqual(outcome.transmitted_codec_bytes, first_scan + 50)
        self.assertEqual(outcome.transmitted_wire_bytes, budget)
        self.assertFalse(outcome.full_stream)

    def test_regular_jpeg_unusable_before_complete_file(self) -> None:
        stream = encode_jpeg(self.image, progressive=False)
        full_wire = jpeg_wire_bytes(len(stream), packet_header_bytes=8)
        short = deliver_jpeg_prefix(
            stream, full_wire - 1, progressive=False, packet_header_bytes=8)
        self.assertIsNone(short.image)
        self.assertLessEqual(short.transmitted_wire_bytes, full_wire - 1)
        self.assertGreater(short.transmitted_codec_bytes, 0)
        full = deliver_jpeg_prefix(
            stream, full_wire, progressive=False, packet_header_bytes=8)
        self.assertEqual(full.image.size, self.image.size)
        self.assertTrue(full.full_stream)

    def test_packet_loss_uses_only_prior_complete_progressive_scan(self) -> None:
        stream = encode_jpeg(self.image, progressive=True)
        first_scan = _scan_end_offsets(stream)[0]
        packet_size = 100
        count = ceil(len(stream) / packet_size)
        lost_index = ceil(first_scan / packet_size)
        self.assertLess(lost_index, count)
        flags = [False] * count
        flags[lost_index] = True
        decoded = decode_received_jpeg(
            stream, loss_flags=flags, progressive=True,
            packet_payload_bytes=packet_size)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.size, self.image.size)
        self.assertIsNone(decode_received_jpeg(
            stream, loss_flags=[True] + [False] * (count - 1),
            progressive=True, packet_payload_bytes=packet_size))
        regular = encode_jpeg(self.image, progressive=False)
        regular_flags = [False] * ceil(len(regular) / packet_size)
        regular_flags[-1] = True
        self.assertIsNone(decode_received_jpeg(
            regular, loss_flags=regular_flags, progressive=False,
            packet_payload_bytes=packet_size))

    def test_optional_quality_search_counts_exact_wire_bytes(self) -> None:
        at_twenty = encode_jpeg(self.image, quality=20)
        budget = jpeg_wire_bytes(len(at_twenty), packet_header_bytes=8)
        stream, meta = search_jpeg_budget(
            self.image, budget, progressive=False, max_quality=30,
            packet_header_bytes=8)
        self.assertEqual(meta["codec_bytes"], len(stream))
        self.assertEqual(meta["wire_bytes"], jpeg_wire_bytes(
            len(stream), packet_header_bytes=8))
        self.assertLessEqual(meta["wire_bytes"], budget)
        self.assertGreaterEqual(meta["jpeg_quality"], 20)
        self.assertEqual(meta["original_size"], [128, 96])
        minimum = encode_jpeg(self.image, quality=1)
        with self.assertRaises(ValueError):
            search_jpeg_budget(
                self.image, jpeg_wire_bytes(len(minimum)) - 1,
                progressive=False)

    def test_classification_adapter_respects_budget_and_prefix(self) -> None:
        progressive = JpegClassificationCodec(
            progressive=True, packet_header_bytes=8)
        full, full_meta = progressive.encode(self.image, budget_bytes=None)
        first_scan = _scan_end_offsets(full)[0]
        budget = jpeg_wire_bytes(first_scan + 120, packet_header_bytes=8)
        prefix, meta = progressive.encode(self.image, budget_bytes=budget)
        self.assertEqual(meta["source_codec_bytes"], full_meta["codec_bytes"])
        self.assertEqual(meta["codec_bytes"], len(prefix))
        self.assertLessEqual(meta["wire_bytes"], budget)
        decoded = progressive.decode(
            prefix, loss_flags=(False,) * ceil(len(prefix) / 1200),
            received_prefix_bytes=None)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.size, self.image.size)
        ordinary = JpegClassificationCodec(
            progressive=False, packet_header_bytes=8)
        q20 = encode_jpeg(self.image, quality=20)
        ordinary_budget = jpeg_wire_bytes(len(q20), packet_header_bytes=8)
        ordinary_stream, ordinary_meta = ordinary.encode(
            self.image, budget_bytes=ordinary_budget)
        self.assertLessEqual(ordinary_meta["wire_bytes"], ordinary_budget)
        self.assertEqual(ordinary_meta["codec_bytes"], len(ordinary_stream))
        self.assertEqual(ordinary_meta["budget_policy"],
                         "complete_file_quality_search")
        self.assertGreaterEqual(ordinary_meta["jpeg_quality"], 20)
        self.assertLess(ordinary_meta["jpeg_quality"], ordinary.quality)
        self.assertTrue(ordinary_stream.endswith(b"\xff\xd9"))
        self.assertIsNotNone(ordinary.decode(
            ordinary_stream,
            loss_flags=(False,) * ceil(len(ordinary_stream) / 1200),
            received_prefix_bytes=None))

    def test_classification_adapter_reports_infeasible_minimum(self) -> None:
        ordinary = JpegClassificationCodec(progressive=False, packet_header_bytes=8)
        quality_one = encode_jpeg(self.image, quality=1)
        minimum_wire = jpeg_wire_bytes(len(quality_one), packet_header_bytes=8)
        stream, meta = ordinary.encode(self.image, budget_bytes=minimum_wire - 1)
        self.assertEqual(stream, b"")
        self.assertEqual(meta["wire_bytes"], 0)
        self.assertIsNone(meta["jpeg_quality"])
        self.assertEqual(meta["status"], "infeasible_quality1")
        self.assertIsNone(ordinary.decode(
            stream, loss_flags=(), received_prefix_bytes=None))

    def test_progressive_exact_scan_end_is_decodable(self) -> None:
        progressive = JpegClassificationCodec(progressive=True)
        full, _ = progressive.encode(self.image, budget_bytes=None)
        first_scan = _scan_end_offsets(full)[0]
        exact, meta = progressive.encode(self.image, budget_bytes=first_scan)
        self.assertEqual(meta["wire_bytes"], first_scan)
        self.assertEqual(len(exact), first_scan)
        self.assertIsNotNone(progressive.decode(
            exact, loss_flags=(False,) * ceil(len(exact) / 1200),
            received_prefix_bytes=None))
        short, _ = progressive.encode(self.image, budget_bytes=first_scan - 1)
        self.assertIsNone(progressive.decode(
            short, loss_flags=(False,) * ceil(len(short) / 1200),
            received_prefix_bytes=None))


if __name__ == "__main__":
    unittest.main()
