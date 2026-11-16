"""正式图 8(a) 只能使用训练集冻结的 30 kB 任务感知参数。"""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replay.evaluation.figure8a_gt_random import (  # noqa: E402
    TARGETS, TRAIN_PROFILE_PROTOCOL, _load_sender_profiles,
)
from replay.transmission.task_jpeg_search import SearchProfile  # noqa: E402


class _Config:
    values = {"test": "fixed"}

    def path(self, value):
        return Path(value)

    def section(self, name):
        return {"image": {"patch_size": 16},
                "importance": {"dct_high_frequency_cutoff": 2}}[name]


def _report() -> dict:
    profile = asdict(SearchProfile(
        8.0, 2.0, True, 2, background_low_frequency_strength=64.0))
    settings = {
        "protocol": TRAIN_PROFILE_PROTOCOL,
        "split": "train", "budget_wire_bytes": 30_000,
        "native_resolution": [2048, 1024], "patch_size": 16,
        "jpeg_restart_markers": False, "background_blur": False,
        "packet_payload_bytes": 1200, "packet_header_bytes": 8,
        "roi_weight": 4.0, "background_weight": 1.0,
        "sample_ids_by_task": {task: [f"train_{task}"] for task in TARGETS},
        "config_sha256": sha256(json.dumps(
            _Config.values, sort_keys=True, default=str).encode()).hexdigest(),
    }
    return {
        "settings": settings,
        "all_train_samples_feasible": True,
        "selected_by_task": {
            task: {"candidate": "444_low64", "profile": profile,
                   "subsampling": 0, "train_feasible_rate": 1.0}
            for task in TARGETS},
    }


class FrozenProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "frozen_profiles.json"
        self.report = _report()

    def load(self, **kwargs):
        self.path.write_text(json.dumps(self.report), encoding="utf-8")
        return _load_sender_profiles(
            _Config(), self.path, budget_bytes=kwargs.get("budget_bytes", 30_000),
            val_image_ids=kwargs.get("val_image_ids", {"val_example"}),
            roi_weight=4.0, background_weight=1.0,
        )

    def test_accepts_frozen_train_profiles_with_hash(self):
        profiles, metadata = self.load()
        self.assertEqual(set(profiles), set(TARGETS))
        self.assertEqual(profiles["car"][0].background_low_frequency_strength, 64.0)
        self.assertEqual(profiles["car"][1], 0)
        self.assertEqual(metadata["sha256"], sha256(self.path.read_bytes()).hexdigest())

    def test_rejects_val_leakage_or_budget_change(self):
        with self.assertRaisesRegex(ValueError, "混入 val"):
            self.load(val_image_ids={"train_car"})
        with self.assertRaisesRegex(ValueError, "30,000"):
            self.load(budget_bytes=31_000)

    def test_rejects_incomplete_train_feasibility(self):
        self.report["all_train_samples_feasible"] = False
        with self.assertRaisesRegex(ValueError, "train 可行"):
            self.load()


if __name__ == "__main__":
    unittest.main()
