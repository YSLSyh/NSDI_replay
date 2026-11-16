"""训练集标定必须完整计入失败样本，并隔离验证集。"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replay.evaluation.tune_gt_30kb_sender import (  # noqa: E402
    Candidate, _candidate_rows, select_train_pairs, summarize_candidates,
)


class TuneGt30KbSenderTests(unittest.TestCase):
    def test_profile_index_zero_is_plain_fallback_not_tuned_candidate(self):
        candidate = Candidate(0, 64.0)
        search = {"candidate_metrics": [
            {"profile_index": 0, "jpeg_quality": 1,
             "codec_wire_bytes": 25_000, "weighted_rgb_mse": 1.0,
             "roi_rgb_mse": 1.0, "background_rgb_mse": 1.0},
            {"profile_index": 1, "jpeg_quality": 10,
             "codec_wire_bytes": 29_000, "weighted_rgb_mse": 8.0,
             "roi_rgb_mse": 3.0, "background_rgb_mse": 13.0},
        ]}
        rows = _candidate_rows(search, (candidate,))
        self.assertEqual(rows[0]["candidate"], candidate.name)
        self.assertEqual(rows[0]["jpeg_quality"], 10)
        self.assertEqual(rows[0]["roi_rgb_mse"], 3.0)

    def test_train_selector_uses_only_train_and_balances_presence(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as scratch:
            root = Path(scratch)
            city = "mockcity"
            for split in ("train", "val"):
                (root / "leftImg8bit" / split / city).mkdir(parents=True)
                (root / "gtFine" / split / city).mkdir(parents=True)
            for name, raw_id in (("car", 26), ("pedestrian", 24),
                                 ("vegetation", 21), ("background", 7)):
                stem = f"{city}_000000_{name}"
                image = Image.new("RGB", (16, 16), (80, 90, 100))
                image.save(root / "leftImg8bit" / "train" / city /
                           f"{stem}_leftImg8bit.png")
                label = np.full((16, 16), raw_id, dtype=np.uint8)
                Image.fromarray(label).save(root / "gtFine" / "train" / city /
                                            f"{stem}_gtFine_labelIds.png")
            # val 文件不能进入标定池。
            stem = f"{city}_000001_val"
            Image.new("RGB", (16, 16)).save(
                root / "leftImg8bit" / "val" / city / f"{stem}_leftImg8bit.png")
            Image.fromarray(np.full((16, 16), 26, np.uint8)).save(
                root / "gtFine" / "val" / city / f"{stem}_gtFine_labelIds.png")
            selected = select_train_pairs(root, positive_per_task=1,
                                          negative_per_task=1, seed=4)
            self.assertEqual(set(selected), {"car", "pedestrian", "vegetation"})
            for task, pairs in selected.items():
                self.assertEqual(len(pairs), 2, task)
                self.assertTrue(all("train" in image.parts for image, _ in pairs))
                self.assertTrue(all("_val" not in image.stem for image, _ in pairs))

    def test_full_feasibility_wins_over_good_roi_on_only_successes(self):
        low = Candidate(0, 16.0)
        high = Candidate(0, 256.0)
        rows = []
        for task in ("car", "pedestrian", "vegetation"):
            for index in range(2):
                rows.append({
                    "task": task, "image_id": f"sample_{index}",
                    "candidates": [
                        {"candidate": low.name, "feasible": index == 0,
                         "roi_rgb_mse": 1.0 if index == 0 else None,
                         "weighted_rgb_mse": 1.0 if index == 0 else None,
                         "wire_bytes": 29_000 if index == 0 else None},
                        {"candidate": high.name, "feasible": True,
                         "roi_rgb_mse": 5.0, "weighted_rgb_mse": 8.0,
                         "wire_bytes": 29_500},
                    ],
                })
        summary, selected, ready = summarize_candidates(rows, (low, high), 2)
        self.assertTrue(ready)
        for task in selected:
            self.assertEqual(selected[task]["candidate"], high.name)
            self.assertEqual(summary[task][low.name]["feasible_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
