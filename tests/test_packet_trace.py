"""真实逐包轨迹必须可追溯、可重复，并拒绝把吞吐量当作丢包。"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from replay.transmission.packet_loss import (
    _packet_loss_events, plan_entropy_packets, simulate_restart_stream_loss,
)
from replay.evaluation.image_experiment import (
    _matched_measured_replays, _measured_trace_fields,
)
from replay.transmission.packet_trace import (
    TraceProvenance, load_packet_trace_csv, load_trace_provenance_json,
)


PROVENANCE = TraceProvenance(
    dataset="measured-5g-example", source_uri="https://example.org/source",
    network="5g_cellular", capture_method="sender/receiver sequence logs",
    loss_evidence="paired_sender_receiver_logs", license="CC-BY-4.0",
)


class PacketTraceTests(unittest.TestCase):
    def test_same_case_uses_same_measured_window_for_different_codecs(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "trace.csv"
            content = "packet_index,timestamp_ms,lost,throughput_mbps\n" + "".join(
                f"{i},{i * 1.25},{int(i in {3, 4, 14, 18})},{25 + i / 10}\n"
                for i in range(30))
            path.write_text(content, encoding="utf-8")
            trace = load_packet_trace_csv(path, provenance=PROVENANCE)
            self.assertEqual(trace.sha256, sha256(path.read_bytes()).hexdigest())
            self.assertEqual(trace.loss_rate, 4 / 30)
            short = trace.select_segment(n_packets=5, window_packets=12,
                                         seed=2026, case_id="frame-001")
            long = trace.select_segment(n_packets=12, window_packets=12,
                                        seed=2026, case_id="frame-001")
            self.assertEqual(short.start_index, long.start_index)
            self.assertEqual(short.loss_trace, long.loss_trace[:5])
            self.assertEqual(short.timestamps_ms, long.timestamps_ms[:5])
            self.assertEqual(short, trace.select_segment(n_packets=5,
                             window_packets=12, seed=2026, case_id="frame-001"))
            self.assertEqual(short.metadata()["source_uri"], PROVENANCE.source_uri)
            self.assertEqual(short.metadata()["trace_sha256"], trace.sha256)
            self.assertTrue(short.metadata()["has_observed_throughput"])
            # 现有 JPEG 分包器直接消费布尔序列，不再调用合成 GE 模型。
            events, model = _packet_loss_events(5, 0.5, 99, "frame-001", "scooter",
                                                "gilbert_elliott", short.loss_trace, 3)
            self.assertEqual(events, short.loss_trace)
            self.assertEqual(model, "external_trace")
            matched = _matched_measured_replays(
                trace, {"jpeg": 5, "progressive_jpeg": 12, "kdc": 8},
                seed=2026, image_id="frame-001", budget=160000,
                scenario="scooter", level="medium")
            self.assertEqual({item.start_index for item in matched.values()},
                             {matched["jpeg"].start_index})
            self.assertEqual(matched["jpeg"].loss_trace,
                             matched["progressive_jpeg"].loss_trace[:5])
            self.assertEqual(matched["kdc"].loss_trace,
                             matched["progressive_jpeg"].loss_trace[:8])
            fields = _measured_trace_fields(matched["jpeg"])
            self.assertEqual(fields["latency_ms"], matched["jpeg"].duration_ms)
            self.assertEqual(fields["latency_kind"],
                             "observed_sender_packet_span_not_end_to_end")
            self.assertIsNone(fields["target_loss_rate"])
            self.assertEqual(fields["trace_sha256"], trace.sha256)
            self.assertEqual(fields["trace_actual_loss_rate"], matched["jpeg"].loss_rate)

    def test_measured_events_drive_real_jpeg_packet_channel(self) -> None:
        image = Image.fromarray(np.random.default_rng(47).integers(
            0, 256, size=(65, 136, 3), dtype=np.uint8), "RGB")
        payloads = {}
        counts = {}
        for method, progressive in (("jpeg", False), ("progressive_jpeg", True)):
            stream = BytesIO()
            image.save(stream, format="JPEG", quality=87, subsampling=0,
                       progressive=progressive,
                       restart_marker_blocks=math.ceil(image.width / 16))
            payloads[method] = stream.getvalue()
            counts[method] = len(plan_entropy_packets(payloads[method], 300)[3])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "measured.csv"
            path.write_text("timestamp_ms,lost\n" + "".join(
                f"{i * 1.5},{int(i % 9 == 2)}\n"
                for i in range(max(counts.values()) + 5)), encoding="utf-8")
            trace = load_packet_trace_csv(path, provenance=PROVENANCE)
            replays = _matched_measured_replays(
                trace, counts, seed=2026, image_id="frame", budget=30000,
                scenario="scooter", level="medium")
            self.assertEqual(len({replay.start_index for replay in replays.values()}), 1)
            for method, payload in payloads.items():
                replay = replays[method]
                outcome = simulate_restart_stream_loss(
                    payload, image_id="frame", scenario="scooter", level="medium",
                    seed=2026, packet_payload_bytes=300, loss_trace=replay.loss_trace)
                self.assertEqual(outcome.lost_packet_indices,
                                 tuple(i for i, lost in enumerate(replay.loss_trace) if lost))
                self.assertEqual(outcome.packet_loss_rate, replay.loss_rate)
                self.assertEqual(outcome.total_packets, len(replay.loss_trace))

    def test_rejects_throughput_only_and_ambiguous_packet_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "trace.csv"
            for content, message in (
                ("timestamp_ms,throughput_mbps\n0,20\n", "仅吞吐量"),
                ("timestamp_ms,lost\n0,\n", "lost"),
                ("timestamp_ms,lost\n0,0\n-1,1\n", "非负数"),
                ("timestamp_ms,lost\n2,0\n1,1\n", "时间戳倒退"),
                ("timestamp_ms,lost\n0,0,extra\n", "列数"),
                ("packet_index,timestamp_ms,lost\n0,0,0\n2,1,1\n", "packet_index"),
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaisesRegex(ValueError, message):
                    load_packet_trace_csv(path, provenance=PROVENANCE)

    def test_no_trace_wrap_or_fabricated_throughput(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "trace.csv"
            path.write_text("timestamp_ms,lost,throughput_mbps\n"
                            "0,0,\n1,1,\n2,0,\n", encoding="utf-8")
            trace = load_packet_trace_csv(path, provenance=PROVENANCE)
            replay = trace.select_segment(n_packets=2, window_packets=3,
                                          seed=1, case_id="frame", start_index=0)
            self.assertEqual(replay.loss_trace, (False, True))
            self.assertEqual(replay.observed_throughput_mbps, (None, None))
            self.assertFalse(replay.metadata()["has_observed_throughput"])
            self.assertIsNone(_measured_trace_fields(replay)["throughput_mbps"])
            with self.assertRaisesRegex(ValueError, "超过"):
                trace.select_segment(n_packets=4, window_packets=4,
                                     seed=1, case_id="frame")
            with self.assertRaisesRegex(ValueError, "start_index"):
                trace.select_segment(n_packets=2, window_packets=3,
                                     seed=1, case_id="frame", start_index=1)

    def test_provenance_sidecar_requires_explicit_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "source.json"
            path.write_text(
                '{"dataset":"measured-5g-example","source_uri":"https://example.org/source",'
                '"network":"5g_cellular","capture_method":"sender/receiver sequence logs",'
                '"loss_evidence":"paired_sender_receiver_logs","license":"CC-BY-4.0"}',
                encoding="utf-8")
            self.assertEqual(load_trace_provenance_json(path), PROVENANCE)
            path.write_text('{"network":"5g_cellular"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "必须且只能"):
                load_trace_provenance_json(path)


if __name__ == "__main__":
    unittest.main()
