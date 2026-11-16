"""分类语义桶：去重训练参考、跨来源多样性和同义拒绝。"""

import unittest

import numpy as np

from replay.evaluation.figure7_bins import (
    CLASSIFICATION_SYSTEM_PROMPT, ReferenceClass, accepted_bin_selection,
    audit_object_families,
    deduplicate_semantic_bins, reaudit_saved_qwen_outputs,
    select_reaudited_attempt,
    select_diverse_references, unique_train_reference_classes,
)


class FakeShared:
    def __init__(self, mapping):
        self.mapping = mapping

    def text(self, prompts):
        return np.asarray([self.mapping[prompt.removeprefix("a photo of ")]
                           for prompt in prompts], dtype=np.float32)


class Figure7BinsTests(unittest.TestCase):
    def test_classification_prompt_requests_parent_families(self) -> None:
        self.assertIn("PARENT OBJECT FAMILIES", CLASSIFICATION_SYSTEM_PROMPT)
        self.assertIn("never a menu of answers", CLASSIFICATION_SYSTEM_PROMPT)
        self.assertIn("several different visible object classes",
                      CLASSIFICATION_SYSTEM_PROMPT)
        self.assertNotIn("List 4–8 specific objects", CLASSIFICATION_SYSTEM_PROMPT)

    def test_many_phone_images_cannot_fill_context(self) -> None:
        labels = {
            "coco_exploratory": ["cell phone", "cat", "truck", "chair", "apple", "guitar"],
            "fss1000": ["smartphone", "dog", "flower", "bicycle", "spoon", "bird"],
        }
        records = []
        for source, names in labels.items():
            for name in names:
                repeats = 100 if name == "cell phone" else 1
                for number in range(repeats):
                    records.append({"split": "train", "source": source,
                                    "has_data": True, "tags": [name],
                                    "id": f"{source}:{name}:{number}"})
        unique = unique_train_reference_classes(records)
        self.assertEqual(len(unique), 12)
        vectors = np.eye(12, dtype=np.float32)
        # 两个手机名称高度相似；多样选择不应同时纳入。
        vectors[6] = vectors[0] * 0.99 + vectors[6] * 0.01
        chosen = select_diverse_references(
            unique, vectors, np.ones(12, np.float32), count=8)
        self.assertEqual(len(chosen), 8)
        self.assertEqual({item.source for item in chosen},
                         {"coco_exploratory", "fss1000"})
        self.assertFalse({"cell phone", "smartphone"}.issubset(
            {item.label for item in chosen}))
        self.assertLessEqual(sum(item.label == "cell phone" for item in chosen), 1)

    def test_phone_synonyms_are_rejected_as_four_semantic_classes(self) -> None:
        mapping = {"cell phone": [1, 0, 0, 0, 0],
                   "phone": [0.99, 0.01, 0, 0, 0],
                   "smartphone": [0.98, 0.02, 0, 0, 0],
                   "mobile device": [0.97, 0.03, 0, 0, 0],
                   "dog": [0, 1, 0, 0, 0],
                   "chair": [0, 0, 1, 0, 0],
                   "flower": [0, 0, 0, 1, 0],
                   "guitar": [0, 0, 0, 0, 1]}
        shared = FakeShared(mapping)
        bad = deduplicate_semantic_bins(
            "cell phone, phone, smartphone, mobile device", shared)
        self.assertEqual(bad["bins"], ["cell phone"])
        self.assertEqual(len(bad["removed_synonyms"]), 3)
        self.assertFalse(accepted_bin_selection(bad))
        good = deduplicate_semantic_bins(
            "cell phone, dog, chair, flower, guitar", shared)
        self.assertEqual(len(good["bins"]), 5)
        self.assertTrue(accepted_bin_selection(good))

    def test_broad_qwen_families_survive_common_clip_similarity(self) -> None:
        # v4 真实诊断中不同上位族共享模板后的余弦为 .85–.90，不能当同义。
        mapping = {"animal": [1.0, 0.0, 0.0, 0.0],
                   "vehicle": [0.862, 0.507, 0.0, 0.0],
                   "furniture": [0.851, 0.0, 0.525, 0.0],
                   "food": [0.895, 0.0, 0.0, 0.446]}
        shared = FakeShared(mapping)
        raw = ("animal, vehicle, furniture, food\n"
               "**Corrected version (after filtering):**\n"
               "animal, vehicle")
        result = deduplicate_semantic_bins(raw, shared)
        self.assertEqual(result["bins"], ["animal", "vehicle", "furniture", "food"])
        self.assertEqual(result["removed_synonyms"], [])
        self.assertTrue(accepted_bin_selection(result))
        self.assertEqual(result["parsed_span"], "animal, vehicle, furniture, food")

    def test_qwen_json_array_is_parsed_without_commentary(self) -> None:
        mapping = {name: np.eye(4)[index] for index, name in enumerate(
            ("animal", "vehicle", "furniture", "plant"))}
        result = deduplicate_semantic_bins(
            '["animal", "vehicle", "furniture", "plant"]\nThese are broad groups.',
            FakeShared(mapping))
        self.assertEqual(result["bins"], list(mapping))

    def test_family_audit_uses_exclusive_support_with_common_clip_component(self) -> None:
        families = ("animal", "vehicle", "furniture", "food")
        base = np.eye(17, dtype=np.float32)
        family_vectors = [0.93 * base[0] + 0.37 * base[index + 1]
                          for index in range(4)]
        candidates = [ReferenceClass("train", f"{family}_{child}",
                                     f"{family}:{child}")
                      for family in families for child in range(3)]
        # 同族子类共享上位特征，但仍是 12 个可区分的训练类。
        vectors = np.asarray([
            0.7 * base[0] + 0.5 * base[1 + index // 3] +
            0.5 * base[5 + index] for index in range(len(candidates))])
        mapping = dict(zip(families, family_vectors))
        result = audit_object_families(
            list(families), FakeShared(mapping), [], candidates, vectors)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["overlapping_families"], [])
        self.assertEqual(result["assignment_policy"],
                         "exclusive_best_bin_with_margin")

    def test_offline_reaudit_uses_saved_qwen_text_without_inventing_bins(self) -> None:
        families = ["animal", "vehicle", "furniture", "food"]
        basis = np.eye(17, dtype=np.float32)
        family_vectors = [0.93 * basis[0] + 0.37 * basis[index + 1]
                          for index in range(4)]
        candidates = [ReferenceClass("train", f"child_{index}", f"id:{index}")
                      for index in range(12)]
        vectors = np.asarray([
            0.7 * basis[0] + 0.5 * basis[1 + index // 3] +
            0.5 * basis[5 + index] for index in range(12)])
        source = {"uses_validation_labels": False, "attempts": [
            {"turn": 1, "raw_model_output": "animal, vehicle, furniture, food"},
            {"turn": 2, "raw_model_output": "vehicle, animal, furniture, food"}]}
        chosen, attempts = reaudit_saved_qwen_outputs(
            source, FakeShared(dict(zip(families, family_vectors))),
            [], candidates, vectors)
        self.assertEqual(chosen, families)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["raw_model_output"],
                         source["attempts"][0]["raw_model_output"])
        self.assertTrue(attempts[0]["accepted"])
        self.assertTrue(attempts[1]["accepted"])
        self.assertEqual(attempts[0]["covered_distinct_train_children"],
                         attempts[1]["covered_distinct_train_children"])

    def test_offline_reaudit_selects_more_families_with_verified_train_support(self) -> None:
        families = ["animal", "vehicle", "furniture", "food",
                    "tool", "plant", "electronic", "building"]
        basis = np.eye(33, dtype=np.float32)
        family_vectors = [0.93 * basis[0] + 0.37 * basis[index + 1]
                          for index in range(8)]
        candidates = [ReferenceClass("train", f"child_{index}", f"id:{index}")
                      for index in range(24)]
        vectors = np.asarray([
            0.7 * basis[0] + 0.5 * basis[1 + index // 3] +
            0.5 * basis[9 + index] for index in range(24)])
        first = ", ".join(families[:4])
        second = ", ".join(families)
        source = {"uses_validation_labels": False, "attempts": [
            {"turn": 1, "raw_model_output": first},
            {"turn": 2, "raw_model_output": second},
            {"turn": 3, "raw_model_output": "animal, vehicle"}]}
        chosen, attempts = reaudit_saved_qwen_outputs(
            source, FakeShared(dict(zip(families, family_vectors))),
            [], candidates, vectors)
        self.assertEqual(chosen, families)
        self.assertEqual(len(attempts), 3)
        self.assertTrue(attempts[0]["accepted"])
        self.assertTrue(attempts[1]["accepted"])
        self.assertFalse(attempts[2]["accepted"])
        self.assertGreater(attempts[1]["covered_distinct_train_children"],
                           attempts[0]["covered_distinct_train_children"])

    def test_reaudit_selection_prefers_family_count_then_coverage_then_order(self) -> None:
        attempts = [
            {"accepted": True, "distinct_count": 4,
             "covered_distinct_train_children": 221},
            {"accepted": True, "distinct_count": 7,
             "covered_distinct_train_children": 205},
            {"accepted": True, "distinct_count": 7,
             "covered_distinct_train_children": 210},
            {"accepted": True, "distinct_count": 7,
             "covered_distinct_train_children": 210},
            {"accepted": False, "distinct_count": 8,
             "covered_distinct_train_children": 999},
        ]
        self.assertEqual(select_reaudited_attempt(attempts), 2)
        attempts[2]["covered_distinct_train_children"] = 205
        attempts[3]["covered_distinct_train_children"] = 205
        self.assertEqual(select_reaudited_attempt(attempts), 1)
        self.assertIsNone(select_reaudited_attempt([attempts[-1]]))

    def test_offline_reaudit_prunes_only_unsupported_model_family(self) -> None:
        supported = ["animal", "vehicle", "furniture", "food",
                     "plant", "electronic", "building"]
        unsupported = "orphan family"
        basis = np.eye(32, dtype=np.float32)
        family_vectors = [0.93 * basis[0] + 0.37 * basis[index + 1]
                          for index in range(7)]
        candidates = [ReferenceClass("train", f"child_{index}", f"id:{index}")
                      for index in range(21)]
        vectors = np.asarray([
            0.7 * basis[0] + 0.5 * basis[1 + index // 3] +
            0.5 * basis[8 + index] for index in range(21)])
        mapping = dict(zip(supported, family_vectors))
        mapping[unsupported] = basis[31]  # 训练参考中无对应子类。
        model_names = supported[:4] + [unsupported] + supported[4:]
        source = {"uses_validation_labels": False, "attempts": [
            {"turn": 1, "raw_model_output": ", ".join(model_names)}]}
        chosen, attempts = reaudit_saved_qwen_outputs(
            source, FakeShared(mapping), [], candidates, vectors)
        self.assertEqual(chosen, supported)
        self.assertEqual(len(attempts), 1)
        attempt = attempts[0]
        self.assertTrue(attempt["accepted"])
        self.assertEqual(attempt["raw_candidates"], model_names)
        self.assertEqual(attempt["original_bins_before_unsupported_filter"], model_names)
        self.assertEqual(attempt["removed_unsupported"][0]["name"], unsupported)
        self.assertEqual(attempt["removed_unsupported"][0]["distinct_train_child_count_before_removal"], 0)
        self.assertEqual(len(attempt["family_audit_rounds"]), 2)
        self.assertFalse(attempt["family_audit_rounds"][0]["audit"]["accepted"])
        self.assertTrue(attempt["family_audit_rounds"][1]["audit"]["accepted"])

    def test_validation_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "train"):
            unique_train_reference_classes([
                {"split": "val", "source": "imagenet", "has_data": True,
                 "tags": ["dog"]}])

    def test_parent_families_need_multiple_distinct_train_children(self) -> None:
        names = ["cat", "dog", "horse", "chair", "table", "bench",
                 "car", "bus", "train", "flower", "tree", "cactus"]
        candidates = [ReferenceClass("train_a" if i % 2 else "train_b", name,
                                     f"record:{i}") for i, name in enumerate(names)]
        label_vectors = np.eye(12, dtype=np.float32)
        group_names = ["animals", "furniture", "vehicles", "plants"]
        mapping = {name: label_vectors[start:start + 3].sum(axis=0)
                   for name, start in zip(group_names, (0, 3, 6, 9))}
        mapping["cat"] = label_vectors[0]
        shared = FakeShared(mapping)
        references = [candidates[index] for index in (0, 3, 6, 9)]
        good = audit_object_families(
            group_names, shared, references, candidates, label_vectors)
        self.assertTrue(good["accepted"])
        self.assertTrue(all(row["distinct_train_child_count"] >= 3
                            for row in good["per_bin"]))
        bad = audit_object_families(
            ["cat", "furniture", "vehicles", "plants"], shared,
            references, candidates, label_vectors)
        self.assertFalse(bad["accepted"])
        self.assertIn("repeats_single_reference_example",
                      bad["per_bin"][0]["reasons"])
        self.assertIn("too_few_distinct_train_child_classes",
                      bad["per_bin"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
