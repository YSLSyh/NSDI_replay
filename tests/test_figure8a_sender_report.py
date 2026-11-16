"""正式 Fig8a 只能读取完整、训练集专属的多维 codec 选择报告。"""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch

from replay.core.config import ReplayConfig
from replay.evaluation.figure8a_dct_ablation import (
    DEFAULT_CANDIDATES, TARGET_RAW_IDS, summarize_dct_rows,
)
from replay.evaluation.figure8a_sender_report import load_dct_sender_candidate
from replay.receiver.cityscapes_packet_adapt import AdaptOptions, selected_sender_codecs


class SenderReportTests(unittest.TestCase):
    def _report(self, config: ReplayConfig) -> dict:
        candidates = [item for item in DEFAULT_CANDIDATES
                      if item["id"] in {"blur16_control", "blur32_420"}]
        sampled = {name: [f"{name}_{number:03d}" for number in range(50)]
                   for name in TARGET_RAW_IDS}
        rows = []
        for name, ids in sampled.items():
            for candidate in candidates:
                for image_id in ids:
                    rows.append({
                        "task": name, "image_id": image_id,
                        "candidate_id": candidate["id"], "status": "ok",
                        "counts": ([8, 1, 1] if candidate["subsampling"] == 2
                                   else [4, 2, 4]),
                        "sent_bytes": 29_000, "jpeg_quality": 32,
                    })
        return {
            "settings": {
                "protocol": "figure8a_train_dct_ablation_v1", "split": "train",
                "pilot": False, "budget_wire_bytes": 30_000,
                "native_resolution": [2048, 1024],
                "jpeg_subsampling": "candidate-dependent 4:4:4 or 4:2:0",
                "optimize_huffman": True,
                "restart_blocks": "candidate-dependent 2048 or 512 MCU",
                "packet_payload_bytes": 1200, "packet_header_bytes": 8,
                "no_loss_codec_only": True,
                "segmenter_revision": "testrev",
                "config_sha256": sha256(json.dumps(
                    config.values, sort_keys=True, default=str).encode()).hexdigest(),
                "sample_images_per_task": 50,
                "sampled_image_ids": sampled,
                "candidates": candidates,
            },
            "runs": rows,
            "summary": summarize_dct_rows(rows, tuple(candidates), 50),
        }

    def test_accepts_complete_train_report_and_selects_420(self) -> None:
        config = ReplayConfig(Path("."), {"models": {"segmenter": {
            "revision": "testrev"}}})
        report = self._report(config)
        train_ids = {name for ids in report["settings"]["sampled_image_ids"].values()
                     for name in ids}
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "artifacts") as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            selected, digest = load_dct_sender_candidate(
                config, path, task="car", budget_bytes=30_000,
                train_image_ids=train_ids)
        self.assertEqual(selected["id"], "blur32_420")
        self.assertEqual(selected["subsampling"], 2)
        self.assertEqual(len(digest), 64)

    def test_rejects_pilot_or_incomplete_training_report(self) -> None:
        config = ReplayConfig(Path("."), {"models": {"segmenter": {
            "revision": "testrev"}}})
        report = self._report(config)
        train_ids = {name for ids in report["settings"]["sampled_image_ids"].values()
                     for name in ids}
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "artifacts") as temporary:
            path = Path(temporary) / "report.json"
            report["settings"]["pilot"] = True
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不匹配"):
                load_dct_sender_candidate(config, path, task="car",
                                          budget_bytes=30_000,
                                          train_image_ids=train_ids)
            report["settings"]["pilot"] = False
            report["runs"].pop()
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未覆盖"):
                load_dct_sender_candidate(config, path, task="car",
                                          budget_bytes=30_000,
                                          train_image_ids=train_ids)

    def test_receiver_adaptation_uses_same_multidimensional_candidate(self) -> None:
        config = ReplayConfig(Path("."), {"models": {"segmenter": {
            "revision": "testrev"}}})
        report = self._report(config)
        train_ids = {name for ids in report["settings"]["sampled_image_ids"].values()
                     for name in ids}
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "artifacts") as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            options = AdaptOptions(
                trace_csv=Path("unused.csv"), trace_provenance=Path("unused.json"),
                output_checkpoint=Path("unused.pt"), device=torch.device("cpu"),
                trace_source_packet_bytes=100, sender_tuning_report=path)
            sigmas, candidates, source = selected_sender_codecs(
                config, options, train_ids)
        self.assertEqual(sigmas["car"], 32.0)
        self.assertEqual(candidates["car"]["subsampling"], 2)
        self.assertEqual(candidates["car"]["restart_blocks"], 512)
        self.assertEqual(source["mode"], "train_tuned_multidimensional")


if __name__ == "__main__":
    unittest.main()
