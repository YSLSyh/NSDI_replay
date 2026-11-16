"""适配接收端全验证集预填：只复用与旧接收端无关的原始逐图行。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from replay.evaluation.figure8a_seed import (SHARED_SETTINGS_KEYS, _source_rows,
                                              seed_adapted_receiver_eval,
                                              validate_seed_for_settings)


class Fig8aSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = [f"val_{number:03d}" for number in range(500)]
        self.methods = ["jpeg", "limitnet_official224", "kdc_direct", "kdc_restored"]
        self.settings = {key: "same" for key in SHARED_SETTINGS_KEYS}
        self.settings.update({
            "protocol": "figure8a_cityscapes_packet_v10_codec_selected",
            "task": "car", "class_id": 13, "seed": 2026,
            "split": "val", "full_validation": True, "pilot": False,
            "budget_wire_bytes": 30_000, "image_ids": self.ids,
            "methods": self.methods, "limitnet_mode": "official224",
            "trace_sha256": "trace", "kdc_codec_candidate": {
                "id": "blur32_420", "subsampling": 2},
            "receiver_checkpoint_sha256": "main_receiver",
        })
        self.rows = []
        for image_id in self.ids:
            for method in self.methods:
                row = {
                    "image_id": image_id, "method": method, "task": "car",
                    "class_id": 13, "budget_wire_bytes": 30_000,
                    "trace_sha256": "trace", "status": "ok", "counts": [1, 0, 0],
                }
                if method == "kdc_restored":
                    row["receiver_applied"] = int(image_id[-3:]) % 20 == 0
                    row["lost_packets"] = int(row["receiver_applied"])
                    row["lost_intervals"] = int(row["receiver_applied"])
                self.rows.append(row)
        self.report = {
            "settings": self.settings,
            "summary": {"task": "car", "class_id": 13,
                        "n_expected_images": 500, "n_complete_images": 500,
                        "all_expected_complete": True},
            "runs": self.rows,
        }
        self.runs_bytes = b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            for row in self.rows)

    def test_seed_preserves_original_lines_and_validates_resume_prefix(self) -> None:
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "artifacts") as temp:
            root = Path(temp)
            source_report = root / "report.json"
            source_runs = root / "runs.jsonl"
            receiver = root / "adapted.pt"
            output = root / "adapted_eval"
            source_report.write_text(json.dumps(self.report), encoding="utf-8")
            source_runs.write_bytes(self.runs_bytes)
            receiver.write_bytes(b"adapted checkpoint")
            provenance = seed_adapted_receiver_eval(
                source_report=source_report, source_runs=source_runs,
                receiver_checkpoint=receiver, output=output,
                task="car", seed=2026)
            self.assertFalse((output / "settings.json").exists())
            self.assertFalse((output / "captured_cases").exists())
            self.assertEqual(provenance["source_rows"], 2000)
            self.assertEqual(provenance["omitted_receiver_applied_restored_rows"], 25)
            self.assertEqual(provenance["seeded_rows"], 1975)
            expected = b"".join(
                line for row, line in zip(self.rows, self.runs_bytes.splitlines(keepends=True))
                if not (row["method"] == "kdc_restored" and row.get("receiver_applied") is True))
            self.assertEqual((output / "runs.jsonl").read_bytes(), expected)
            self.assertEqual(provenance["seeded_prefix_sha256"], sha256(expected).hexdigest())
            adapted_settings = {key: self.settings[key] for key in SHARED_SETTINGS_KEYS}
            adapted_settings.update({
                "protocol": "figure8a_cityscapes_packet_v10_codec_selected_receiver_adapted",
                "case_capture": "disabled_explicit_adapted_ablation",
                "receiver_checkpoint_sha256": sha256(receiver.read_bytes()).hexdigest(),
                "receiver_adaptation_metadata": {
                    "initial_receiver_sha256": "main_receiver"},
            })
            validate_seed_for_settings(output, adapted_settings)
            wrong_trace = {**adapted_settings, "trace_sha256": "other trace"}
            with self.assertRaisesRegex(ValueError, "设置不一致"):
                validate_seed_for_settings(output, wrong_trace)
            mask_only = {**adapted_settings,
                         "receiver_output_policy": "replace_missing_entropy_intervals_only_no_halo"}
            with self.assertRaisesRegex(ValueError, "输出策略不一致"):
                validate_seed_for_settings(output, mask_only)
            mask_output = root / "mask_only_eval"
            mask_provenance = seed_adapted_receiver_eval(
                source_report=source_report, source_runs=source_runs,
                receiver_checkpoint=receiver, output=mask_output,
                task="car", seed=2026, receiver_mask_only=True)
            self.assertEqual(mask_provenance["receiver_output_mode"], "mask_only")
            self.assertEqual((mask_output / "runs.jsonl").read_bytes(), expected)
            mask_settings = {
                **mask_only,
                "protocol": ("figure8a_cityscapes_packet_v10_codec_selected_"
                             "receiver_adapted_mask_only"),
            }
            validate_seed_for_settings(mask_output, mask_settings)
            with self.assertRaisesRegex(ValueError, "输出策略不一致"):
                validate_seed_for_settings(mask_output, adapted_settings)
            with (output / "runs.jsonl").open("ab") as destination:
                destination.write(b"{\"new\":true}\n")
            validate_seed_for_settings(output, adapted_settings)
            changed = (output / "runs.jsonl").read_bytes()
            (output / "runs.jsonl").write_bytes(b"X" + changed[1:])
            with self.assertRaisesRegex(ValueError, "字节前缀"):
                validate_seed_for_settings(output, adapted_settings)

    def test_rejects_incomplete_or_wrong_source(self) -> None:
        wrong = json.loads(json.dumps(self.report))
        wrong["settings"]["seed"] = 123
        with self.assertRaisesRegex(ValueError, "同 seed"):
            _source_rows(wrong, self.runs_bytes, task="car", seed=2026)
        wrong = json.loads(json.dumps(self.report))
        wrong["runs"].pop()
        with self.assertRaisesRegex(ValueError, "完整逐图记录"):
            _source_rows(wrong, self.runs_bytes, task="car", seed=2026)
        wrong = json.loads(json.dumps(self.report))
        wrong["settings"]["image_ids"][-1] = wrong["settings"]["image_ids"][0]
        with self.assertRaisesRegex(ValueError, "唯一"):
            _source_rows(wrong, self.runs_bytes, task="car", seed=2026)


if __name__ == "__main__":
    unittest.main()
