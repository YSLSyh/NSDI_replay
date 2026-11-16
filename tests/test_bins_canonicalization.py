"""验证自动驾驶语义桶只来自模型候选，且同义词不会造成类别缺失。"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from replay.data.cityscapes import TRAIN_NAMES
from replay.knowledge.bins import select_required_bins, select_required_bins_with_retry


REQUIRED = ["car", "pedestrian", "traffic light", "traffic sign"]
TASK = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" /
                       "tx_stage12_qwen4b.yaml").read_text(encoding="utf-8"))["task"]
ALIASES = TASK["bin_aliases"]
ALLOWED = TASK["allowed_bins"]


class BinSelectionTest(unittest.TestCase):
    def test_config_retains_paper_range_and_evaluable_optional_bins(self) -> None:
        self.assertEqual((TASK["bin_min"], TASK["bin_max"]), (4, 8))
        self.assertEqual(TASK["required_bins"], REQUIRED)
        self.assertGreater(len(ALLOWED), len(REQUIRED))
        self.assertEqual(len(ALLOWED), 8)
        self.assertTrue(set(ALLOWED) - set(REQUIRED))
        self.assertTrue({"person" if name == "pedestrian" else name
                         for name in ALLOWED}.issubset(TRAIN_NAMES))

    def test_strict_equivalent_alias_families(self) -> None:
        for answer in (
            "CAR, pedestrian, traffic light, traffic sign",
            "MotorCar, Person, Stop-Light, Street_Sign",
            "AUTOMOBILES, PEDESTRIANS, Traffic_Signals, Road Signs",
            "cars, people, stoplights, street signs",
            "motorcars, persons, stop lights, road signs",
        ):
            with self.subTest(answer=answer):
                result = select_required_bins(answer, REQUIRED, ALLOWED, ALIASES)
                self.assertEqual(result["selected_bins"], REQUIRED)
                self.assertEqual(result["missing_required_bins"], [])

    def test_aliases_are_kept_and_unrelated_candidates_excluded(self) -> None:
        result = select_required_bins(
            "automobile, pedestrians, traffic signal, road sign, road, building",
            REQUIRED, ALLOWED, ALIASES)
        self.assertEqual(result["selected_bins"], REQUIRED)
        self.assertEqual(result["excluded_candidates"], ["road", "building"])
        self.assertEqual(result["missing_required_bins"], [])

    def test_missing_class_is_not_added(self) -> None:
        result = select_required_bins(
            "car, person, traffic signal, road, building", REQUIRED, ALLOWED, ALIASES)
        self.assertEqual(result["missing_required_bins"], ["traffic sign"])
        self.assertNotIn("traffic sign", result["selected_bins"])

    def test_vehicle_aliases_map_to_car_and_optional_classes_remain_distinct(self) -> None:
        result = select_required_bins(
            "truck, bus, vehicle, road, building, pedestrian, traffic light, traffic sign, rider, bike",
            REQUIRED, ALLOWED, ALIASES)
        self.assertEqual(result["missing_required_bins"], [])
        self.assertEqual(result["selected_bins"], REQUIRED + ["rider", "bicycle"])
        self.assertEqual(result["excluded_candidates"],
                         ["vehicle", "road", "building"])

    def test_case_space_separator_and_alias_duplicates_are_removed(self) -> None:
        result = select_required_bins(
            " CAR, car, automobile, motorcar, PEDESTRIAN, pedestrians, "
            "Traffic-Light, TRAFFIC__LIGHT, stop light, road_sign, traffic sign ",
            REQUIRED, ALLOWED, ALIASES)
        self.assertEqual(result["raw_candidates"],
                         ["car", "automobile", "motorcar", "pedestrian", "pedestrians",
                          "traffic light", "stop light", "road sign", "traffic sign"])
        self.assertEqual(result["canonical_candidates"], REQUIRED)
        self.assertEqual(result["selected_bins"], REQUIRED)

    def test_out_of_range_retries_once_and_audits_both_outputs(self) -> None:
        called = []
        initial = "car, pedestrian, traffic light"
        second = "motorcar, person, stoplight, street sign, road"

        def retry(previous: str) -> str:
            called.append(previous)
            return second

        result = select_required_bins_with_retry(
            initial, 4, 8, REQUIRED, ALLOWED, ALIASES, retry)
        self.assertEqual(called, [initial])
        self.assertEqual(result["raw_model_outputs"], [initial, second])
        self.assertEqual([attempt["raw_count"] for attempt in result["attempts"]], [3, 5])
        self.assertTrue(result["raw_count_in_paper_range"])
        self.assertEqual(result["selection"]["selected_bins"], REQUIRED)

    def test_retry_cannot_fill_a_missing_required_class_by_code(self) -> None:
        initial = "car, pedestrian, traffic light"
        second = "car, pedestrian, traffic light, road"
        result = select_required_bins_with_retry(
            initial, 4, 8, REQUIRED, ALLOWED, ALIASES, lambda _: second)
        self.assertEqual(result["selection"]["missing_required_bins"], ["traffic sign"])
        self.assertEqual(result["selection"]["selected_bins"], REQUIRED[:-1])

    def test_valid_raw_count_retries_when_required_class_missing(self) -> None:
        called = []
        second = "car, pedestrian, traffic light, traffic sign, motorcycle"
        result = select_required_bins_with_retry(
            "car, pedestrian, traffic light, road", 4, 8, REQUIRED, ALLOWED,
            ALIASES, lambda previous: called.append(previous) or second)
        self.assertEqual(len(called), 1)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["selection"]["selected_bins"], REQUIRED + ["motorcycle"])
        self.assertTrue(result["accepted"])

    def test_selected_count_below_four_retries_even_when_raw_count_is_valid(self) -> None:
        result = select_required_bins_with_retry(
            "car, automobile, person, traffic light, road", 4, 8,
            REQUIRED, ALLOWED, ALIASES,
            lambda _: "car, pedestrian, traffic light, traffic sign, bicycle, motorcycle")
        self.assertEqual(result["attempts"][0]["selected_count"], 3)
        self.assertFalse(result["attempts"][0]["selected_count_in_paper_range"])
        self.assertEqual(result["selection"]["selected_bins"],
                         REQUIRED + ["motorcycle", "bicycle"])
        self.assertTrue(result["accepted"])


if __name__ == "__main__":
    unittest.main()
