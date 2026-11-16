import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from replay.evaluation.figure8 import binary_counts
from replay.evaluation.figure8a_capture import (CAPTURE_PROTOCOL,
                                                 ExactCaseCapture,
                                                 capture_role_scores)
from replay.evaluation.figure8a_packet import _image_sha256
from replay.evaluation.figure8a_visuals import _render_exact_capture


class ExactCaptureTest(unittest.TestCase):
    def test_lossy_roles_require_real_kdc_loss_and_report_signed_gain(self) -> None:
        rows = {
            "jpeg": {"status": "ok", "counts": [6, 0, 4]},
            "limitnet_official224": {"status": "ok", "counts": [7, 0, 3]},
            "kdc_direct": {"status": "ok", "counts": [8, 0, 2],
                           "lost_packets": 1},
            "kdc_restored": {"status": "ok", "counts": [7, 0, 3],
                             "gt_target_pixels": 10, "receiver_applied": True},
        }
        scores = capture_role_scores(rows, "limitnet_official224")
        self.assertAlmostEqual(scores["best_lossy_kdc_vs_jpeg"], 0.1)
        self.assertAlmostEqual(scores["best_restoration_gain"], -0.1)
        rows["kdc_direct"]["lost_packets"] = 0
        no_loss = capture_role_scores(rows, "limitnet_official224")
        self.assertNotIn("best_lossy_kdc_vs_jpeg", no_loss)
        self.assertNotIn("best_restoration_gain", no_loss)
        rows["kdc_direct"]["lost_packets"] = 1
        rows["kdc_restored"]["receiver_applied"] = False
        self.assertNotIn("best_restoration_gain",
                         capture_role_scores(rows, "limitnet_official224"))

    def test_online_capture_and_model_free_verified_render(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "artifacts") as directory:
            root = Path(directory)
            truth = np.zeros((16, 32), dtype=np.uint8)
            truth[:, :16] = 13
            original = Image.new("RGB", (32, 16), (70, 80, 90))
            original.save(root / "a.png")
            original.save(root / "b.png")
            original.save(root / "c.png")
            limit_method = "limitnet_official224"
            methods = ["jpeg", limit_method, "kdc_direct", "kdc_restored"]
            settings = {
                "task": "car", "class_id": 13,
                "seed": 2026,
                "trace_sha256": "trace", "config_sha256": "config",
                "budget_wire_bytes": 30000,
                "limitnet_checkpoint_sha256": "limitnet",
                "receiver_checkpoint_sha256": "receiver",
                "methods": methods, "image_ids": ["a", "b", "c"],
                "full_validation": False, "pilot": True,
                "capture_protocol": CAPTURE_PROTOCOL,
                "capture_manifest": "captured_cases/manifest.json",
                "packet_payload_bytes": 1200,
                "limitnet_mode": "official224",
            }
            manager = ExactCaseCapture(root, settings, existing_rows=0)
            all_rows = []
            for image_id in ("a", "b", "c"):
                pictures = {method: Image.new("RGB", (32, 16), (i * 40, 30, 50))
                            for i, method in enumerate(("jpeg", "limitnet",
                                                        "kdc_direct", "kdc_restored"))}
                masks = {method: truth.copy() for method in pictures}
                masks["jpeg"][:, 12:16] = 0
                masks["limitnet"][:, 10:16] = 0
                if image_id == "b":
                    masks["kdc_restored"][:, 8:16] = 0
                if image_id == "c":
                    masks["jpeg"][:, 4:16] = 0
                    masks["kdc_restored"][:, 12:16] = 0
                rows = {}
                for method in pictures:
                    report_method = limit_method if method == "limitnet" else method
                    row = {"image_id": image_id, "method": report_method,
                           "status": "ok", "gt_target_pixels": int((truth == 13).sum()),
                           "counts": list(binary_counts(masks[method], truth, 13)),
                           "decoded_image_sha256": _image_sha256(pictures[method]),
                           "trace_start_index": 3,
                           "lost_packets": (2 if image_id == "c" and
                                             method in {"kdc_direct", "kdc_restored"}
                                             else 0),
                           "receiver_applied": (image_id == "c" and
                                                method == "kdc_restored")}
                    if method == "kdc_restored":
                        row["restored_image_sha256"] = _image_sha256(pictures[method])
                    rows[report_method] = row
                    all_rows.append(row)
                manager.consider(
                    image_id=image_id, original=original, truth=truth,
                    missing=np.zeros_like(truth), pictures=pictures,
                    predictions=masks, rows=rows)
            capture = json.loads((root / "captured_cases/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(capture["roles"]),
                             {"best_vs_limitnet", "worst_vs_limitnet", "best_vs_jpeg",
                              "best_lossy_kdc_vs_jpeg", "best_restoration_gain"})
            report = {"settings": settings, "runs": all_rows}
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            trace = SimpleNamespace(
                sha256="trace", provenance=SimpleNamespace(
                    capture_method="source_packet_bytes=100", dataset="5G test",
                    network="5G SA"))
            config = SimpleNamespace(data_path=lambda _: root)
            with (patch("replay.evaluation.figure8a_visuals.cityscapes_pairs",
                        return_value=[(root / "a.png", root / "truth.png"),
                                      (root / "b.png", root / "truth.png"),
                                      (root / "c.png", root / "truth.png")]),
                  patch("replay.evaluation.figure8a_visuals.load_cityscapes_pair",
                        side_effect=lambda *_: (original.copy(), truth.copy()))):
                result = _render_exact_capture(
                    report, report_file=report_path, config=config,
                    trace=trace, output=root / "visuals", allow_pilot=True)
            self.assertEqual(len(result["examples"]), 5)
            self.assertFalse(result["codec_or_segmenter_rerun"])
            restored = next(case for case in result["examples"]
                            if case["role"] == "best_restoration_gain")
            self.assertEqual(restored["observed_kdc_lost_packets"], 2)
            self.assertTrue(restored["receiver_applied"])
            self.assertFalse(restored["positive_gain"])
            self.assertTrue(all(Path(item["preview"]).is_file()
                                for item in result["examples"]))


if __name__ == "__main__":
    unittest.main()
