"""图 7 的完整验证集和同信道配对审计。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from replay.evaluation.figure7_classification import (
    BUDGET_METHODS, LOSS_METHODS, ValSample, evaluate_codec_point,
    summarize_figure7,
    validate_imagenet_val, validate_intact_predictions,
)


class Figure7ClassificationTests(unittest.TestCase):
    _TEMP_PARENT = Path(__file__).resolve().parents[1] / "artifacts"

    def test_val_layout_and_wnid_index_order(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            root = Path(temp)
            for wnid in ("n01443537", "n01440764"):
                folder = root / wnid
                folder.mkdir()
                for number in range(2):
                    (folder / f"val_{number}.JPEG").write_bytes(b"jpeg fixture")
            samples, digest = validate_imagenet_val(
                root, expected_classes=2, expected_images=4, expected_per_class=2)
            self.assertEqual(len(samples), 4)
            self.assertEqual([item.target for item in samples], [0, 0, 1, 1])
            self.assertEqual(len(digest), 64)
            (root / "n01440764" / "val_2.JPEG").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "恰好"):
                validate_imagenet_val(
                    root, expected_classes=2, expected_images=4, expected_per_class=2)

    def _fixture(self, root: Path) -> tuple[list[ValSample], Path, Path]:
        samples = [ValSample(f"n01440764/ILSVRC2012_val_{index+1:08d}.JPEG",
                             root / f"sample{index}.JPEG", index % 2,
                             "n01440764") for index in range(4)]
        intact = root / "intact.jsonl"
        intact.write_text("".join(json.dumps({
            "image_id": sample.image_id, "target": sample.target,
            "prediction": sample.target,
            "backend": "torchvision_vit_l_16_swag_e2e",
            "weights_sha256": "a" * 64}) + "\n" for sample in samples),
            encoding="utf-8")
        settings = {
            "dataset_image_count": 4, "dataset_manifest_sha256": "fixture",
            "classification_metric": "top1", "split": "val",
            "classifier_is_paper_specified": False,
            "classifier": {"backend": "torchvision_vit_l_16_swag_e2e",
                           "weights_sha256": "a" * 64},
            "intact_predictions_path": intact.name,
            "loss_rates": [0.1], "budgets_bytes": [30_000],
            "loss_methods": list(LOSS_METHODS),
            "budget_methods": list(BUDGET_METHODS),
            "channel": {"packet_payload_bytes": 1200, "random_seed": 7},
        }
        settings_path = root / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        trials_path = root / "trials.jsonl"
        rows = []
        for axis, x, methods in (("loss", 0.1, LOSS_METHODS),
                                 ("budget", 30_000, BUDGET_METHODS)):
            for method in methods:
                for sample in samples:
                    row = {
                        "axis": axis, "x": x, "method": method,
                        "image_id": sample.image_id, "target": sample.target,
                        "prediction": -1 if method == "jpeg" else sample.target,
                        "sent_bytes": 1200, "channel_event_id":
                            f"seed7:{axis}:{x}:{sample.image_id}",
                        "receiver_session_update": False if method == "kdc" else None,
                    }
                    if method in ("jpeg", "prog_jpeg"):
                        row.update({"jpeg_quality": (20 + sample.target
                                                     if method == "jpeg" else 75),
                                    "codec_bytes": 1192, "wire_bytes": 1200})
                    rows.append(row)
        trials_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                               encoding="utf-8")
        return samples, settings_path, trials_path

    def test_intact_gate_rejects_mislabeled_or_low_accuracy(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            samples, _, _ = self._fixture(Path(temp))
            path = Path(temp) / "intact.jsonl"
            result = validate_intact_predictions(path, samples)
            self.assertEqual(result["top1"], 1.0)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["prediction"] = 999
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "低于"):
                validate_intact_predictions(path, samples)

    def test_report_requires_complete_paired_trials(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            root = Path(temp)
            samples, settings, trials = self._fixture(root)
            with patch("replay.evaluation.figure7_classification.validate_imagenet_val",
                       return_value=(samples, "fixture")), patch(
                       "replay.evaluation.figure7_classification.EXPECTED_IMAGES", 4):
                report = summarize_figure7(
                    val_root=root, settings_path=settings,
                    trials_path=trials, output_dir=root / "out")
                self.assertEqual(report["intact"]["top1"], 1.0)
                self.assertEqual(report["curves"]["budget"]["jpeg"][0]["top1"], 0.0)
                wire = report["curves"]["loss"]["prog_jpeg"][0]
                self.assertEqual((wire["mean_sent_bytes"], wire["median_sent_bytes"],
                                  wire["min_sent_bytes"], wire["max_sent_bytes"]),
                                 (1200, 1200, 1200, 1200))
                self.assertEqual(wire["jpeg_quality_range"], [75, 75])
                self.assertEqual(wire["mean_codec_bytes"], 1192)
                self.assertEqual(report["curves"]["budget"]["jpeg"][0]
                                 ["jpeg_quality_range"], [20, 21])
                self.assertIn("does not enforce equal transmitted bytes",
                              report["loss_axis_rate_policy"])
                self.assertTrue((root / "out" / "figure7_loss.png").is_file())
                rows = [json.loads(line) for line in trials.read_text().splitlines()]
                rows[0]["channel_event_id"] = "different loss pattern"
                trials.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "配对信道事件"):
                    summarize_figure7(val_root=root, settings_path=settings,
                                      trials_path=trials, output_dir=root / "bad")
                self.assertFalse((root / "bad").exists())

    def test_report_rejects_kdc_cross_image_session_update(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            root = Path(temp)
            samples, settings, trials = self._fixture(root)
            rows = [json.loads(line) for line in trials.read_text().splitlines()]
            next(row for row in rows if row["method"] == "kdc")["receiver_session_update"] = True
            trials.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch("replay.evaluation.figure7_classification.validate_imagenet_val",
                       return_value=(samples, "fixture")), patch(
                       "replay.evaluation.figure7_classification.EXPECTED_IMAGES", 4):
                with self.assertRaisesRegex(ValueError, "逐图 Top-1"):
                    summarize_figure7(val_root=root, settings_path=settings,
                                      trials_path=trials, output_dir=root / "bad")

    def test_report_rejects_inconsistent_jpeg_byte_ledger(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            root = Path(temp)
            samples, settings, trials = self._fixture(root)
            rows = [json.loads(line) for line in trials.read_text().splitlines()]
            next(row for row in rows if row["method"] == "jpeg")["wire_bytes"] = 1199
            trials.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch("replay.evaluation.figure7_classification.validate_imagenet_val",
                       return_value=(samples, "fixture")), patch(
                       "replay.evaluation.figure7_classification.EXPECTED_IMAGES", 4):
                with self.assertRaisesRegex(ValueError, "JPEG 逐图"):
                    summarize_figure7(val_root=root, settings_path=settings,
                                      trials_path=trials, output_dir=root / "bad")

    def test_evaluator_pairs_packet_randomness_across_codecs(self) -> None:
        class FakeCodec:
            def __init__(self, name: str, size: int):
                self.name, self.size = name, size
                self.loss_flags = None

            def encode(self, image, *, budget_bytes):
                size = self.size if budget_bytes is None else min(self.size, budget_bytes)
                return b"x" * size, {"wire_bytes": size, "codec_bytes": size,
                                    "jpeg_quality": 75}

            def decode(self, payload, *, loss_flags, received_prefix_bytes):
                self.loss_flags = loss_flags
                return Image.new("RGB", (8, 8))

        class FakeClassifier:
            def predict(self, image):
                return 0

        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            root = Path(temp)
            image_path = root / "sample.JPEG"
            Image.new("RGB", (8, 8)).save(image_path)
            sample = ValSample("n01440764/one.JPEG", image_path, 0, "n01440764")
            short = FakeCodec("jpeg", 1000)
            long = FakeCodec("prog_jpeg", 2500)
            rows = evaluate_codec_point(
                samples=[sample], classifier=FakeClassifier(),
                codecs={"jpeg": short, "prog_jpeg": long},
                axis="loss", x=0.2, loss_rate=0.5,
                packet_payload_bytes=1000, channel_seed=42)
            self.assertEqual(rows[0]["channel_event_id"], rows[1]["channel_event_id"])
            self.assertEqual(short.loss_flags, long.loss_flags[:1])
            self.assertEqual(rows[0]["prediction"], 0)
            self.assertEqual(rows[0]["jpeg_quality"], 75)
            self.assertEqual(rows[0]["codec_bytes"], 1000)
            self.assertEqual(rows[0]["wire_bytes"], 1000)

    def test_evaluator_rejects_invalid_jpeg_quality(self) -> None:
        class BadCodec:
            name = "jpeg"

            def encode(self, image, *, budget_bytes):
                return b"jpeg", {"wire_bytes": 4, "codec_bytes": 4,
                                 "jpeg_quality": True}

            def decode(self, payload, *, loss_flags, received_prefix_bytes):
                raise AssertionError("无效质量应在解码前被拒绝")

        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            image_path = Path(temp) / "sample.JPEG"
            Image.new("RGB", (8, 8)).save(image_path)
            sample = ValSample("n01440764/one.JPEG", image_path, 0, "n01440764")
            with self.assertRaisesRegex(ValueError, "JPEG 质量"):
                evaluate_codec_point(
                    samples=[sample], classifier=object(), codecs={"jpeg": BadCodec()},
                    axis="budget", x=100, loss_rate=0, packet_payload_bytes=1200,
                    channel_seed=42)

    def test_report_accepts_complete_available_method_subset(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temp:
            root = Path(temp)
            samples, settings_path, trials_path = self._fixture(root)
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            settings["loss_methods"] = ["prog_jpeg", "kdc"]
            settings["budget_methods"] = ["jpeg", "prog_jpeg", "kdc"]
            settings_path.write_text(json.dumps(settings), encoding="utf-8")
            rows = [json.loads(line) for line in trials_path.read_text().splitlines()]
            rows = [row for row in rows if row["method"] in settings[f"{row['axis']}_methods"]]
            trials_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch("replay.evaluation.figure7_classification.validate_imagenet_val",
                       return_value=(samples, "fixture")), patch(
                       "replay.evaluation.figure7_classification.EXPECTED_IMAGES", 4):
                report = summarize_figure7(
                    val_root=root, settings_path=settings_path,
                    trials_path=trials_path, output_dir=root / "subset")
            self.assertEqual(report["methods_unavailable"]["loss"],
                             ["limitnet", "starfish"])
            self.assertEqual(report["methods_unavailable"]["budget"],
                             ["limitnet", "starfish"])


if __name__ == "__main__":
    unittest.main()
