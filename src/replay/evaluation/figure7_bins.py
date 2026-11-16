"""ImageNet 分类任务的多样训练参考和同义桶审计。

只使用 PosteriorKB 的 train 图像标签；不查看 ImageNet val 的文件名或真值。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from typing import Any

import numpy as np

from replay.knowledge.bins import _parse_candidates


CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an expert in visual scene analysis preparing semantic bins for a "
    "general image classification task. Retrieved training labels are only "
    "examples of the data distribution, never a menu of answers. Return 4–8 "
    "concrete, visually localizable PARENT OBJECT FAMILIES. Each family must "
    "contain several different visible object classes, not synonyms or subtypes "
    "of a single class. Cover the broad task rather than random examples. "
    "Avoid vague concepts such as things, image, scene, and background. "
    "Return only comma-separated family names."
)


@dataclass(frozen=True)
class ReferenceClass:
    source: str
    label: str
    record_id: str


def _normalized(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.strip().casefold()).strip()


def _parse_classification_candidates(response: str) -> tuple[list[str], str]:
    """从 Qwen 原文取第一个明确列表；后续自我纠错文字不算类别。"""
    for line in response.splitlines():
        stripped = line.strip().strip("` ")
        if not stripped:
            continue
        if stripped.startswith("["):
            try:
                value, _ = json.JSONDecoder().raw_decode(stripped)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    return list(dict.fromkeys(_normalized(item) for item in value
                                              if _normalized(item))), stripped
        if stripped.count(",") + stripped.count("，") >= 3:
            candidate_line = stripped.split(":", 1)[-1] if ":" in stripped else stripped
            return _parse_candidates(candidate_line), stripped
    return _parse_candidates(response), response


def _unit_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("CLIP 文本嵌入维度或数值无效")
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("CLIP 文本嵌入包含零向量")
    return array / norm


def unique_train_reference_classes(records: list[dict[str, Any]]) -> list[ReferenceClass]:
    """同来源同标签无论多少图像只保留一个候选；非 train 数据一律拒绝。"""
    if not records or any(not isinstance(row, dict) or row.get("split") != "train"
                          for row in records):
        raise ValueError("图 7 分类参考 KB 必须只含 train 记录")
    seen: set[tuple[str, str]] = set()
    result = []
    for row in records:
        if row.get("has_data") is not True:
            continue
        tags = row.get("tags")
        if not isinstance(tags, list) or len(tags) != 1 or not isinstance(tags[0], str):
            continue
        source = str(row.get("source", "unknown"))
        label = _normalized(tags[0])
        if not label or (source, label) in seen:
            continue
        seen.add((source, label))
        result.append(ReferenceClass(source, label, str(row.get("id", ""))))
    if len(result) < 8:
        raise ValueError("train-only KB 中不足 8 个不同的图像参考类别")
    return result


def select_diverse_references(
    candidates: list[ReferenceClass], vectors: np.ndarray,
    task_vector: np.ndarray, *, count: int = 32,
    near_duplicate_cosine: float = 0.84,
) -> list[ReferenceClass]:
    """按来源轮询 + MMR 新颖性，防止 top-k 被同类近重复占满。"""
    if not candidates or count <= 0 or not 0 < near_duplicate_cosine < 1:
        raise ValueError("多样参考候选或阈值无效")
    embeddings = _unit_rows(vectors)
    if embeddings.shape[0] != len(candidates):
        raise ValueError("参考类别与文本嵌入数量不一致")
    task = _unit_rows(np.asarray(task_vector, dtype=np.float32).reshape(1, -1))[0]
    if embeddings.shape[1] != task.size:
        raise ValueError("任务与参考类别嵌入维度不一致")
    sources = sorted({item.source for item in candidates})
    relevance = embeddings @ task
    selected: list[int] = []
    remaining = set(range(len(candidates)))
    quota = math.ceil(count / len(sources))
    source_counts: Counter[str] = Counter()
    while len(selected) < count:
        progress = False
        for source in sources:
            if len(selected) >= count:
                break
            if source_counts[source] >= quota:
                continue
            eligible = []
            for index in remaining:
                if candidates[index].source != source:
                    continue
                novelty = 1.0
                if selected:
                    similarity = float(np.max(embeddings[selected] @ embeddings[index]))
                    if similarity >= near_duplicate_cosine:
                        continue
                    novelty = 1.0 - similarity
                # 任务相关性与语义新颖性共同决定候选，避免单一热门类占满上下文。
                score = 0.35 * float(relevance[index]) + 0.65 * novelty
                eligible.append((score, index))
            if not eligible:
                continue
            index = max(eligible, key=lambda item: (item[0], -item[1]))[1]
            selected.append(index)
            remaining.remove(index)
            source_counts[source] += 1
            progress = True
        if not progress:
            break
    if len(selected) < min(count, 8):
        raise ValueError("KB 无法提供至少 8 个跨语义的训练参考类别")
    return [candidates[index] for index in selected]


def deduplicate_semantic_bins(
    response: str, shared: Any, *, semantic_cosine: float = 0.94,
) -> dict[str, Any]:
    """仅高置信同义合并；上位对象族不可因 CLIP 公共模板而误合并。"""
    raw, parsed_span = _parse_classification_candidates(response)
    if not raw:
        return {"raw_candidates": [], "bins": [], "removed_synonyms": [],
                "parsed_span": parsed_span}
    # 只做文本相似度时，公共的 "a photo of" 会把不同上位族推到 .85+。
    vectors = _unit_rows(shared.text(raw))
    kept: list[int] = []
    removed: list[dict[str, Any]] = []
    for index, name in enumerate(raw):
        matched = None
        for previous in kept:
            other = raw[previous]
            similarity = float(vectors[index] @ vectors[previous])
            tokens_a, tokens_b = set(name.split()), set(other.split())
            lexical_family = bool(tokens_a and tokens_b and
                                  (tokens_a <= tokens_b or tokens_b <= tokens_a)
                                  and similarity >= 0.68)
            embedded_word = (len(name) >= 4 and len(other) >= 4 and
                             (name in other or other in name) and similarity >= 0.75)
            if similarity >= semantic_cosine or lexical_family or embedded_word:
                matched = (other, similarity)
                break
        if matched is None:
            kept.append(index)
        else:
            removed.append({"name": name, "same_as": matched[0],
                            "clip_cosine": matched[1]})
    return {"raw_candidates": raw, "bins": [raw[index] for index in kept],
            "removed_synonyms": removed,
            "parsed_span": parsed_span,
            "semantic_cosine_threshold": semantic_cosine}


def accepted_bin_selection(selection: dict[str, Any]) -> bool:
    """原始输出和同义过滤结果都必须处于论文的 4–8 类范围。"""
    return (4 <= len(selection["raw_candidates"]) <= 8 and
            4 <= len(selection["bins"]) <= 8)


def audit_object_families(
    bins: list[str], shared: Any, references: list[ReferenceClass],
    candidate_classes: list[ReferenceClass], candidate_vectors: np.ndarray,
    *, min_distinct_children: int = 3, assignment_margin: float = 0.02,
) -> dict[str, Any]:
    """每个上位桶须覆盖多个不同的 train 类，不得只复述 RAG 中的实例。

    这是分类任务的保守验收门槛，不会替 Qwen 增加或替换任何桶。
    """
    if len(candidate_classes) != len(candidate_vectors):
        raise ValueError("分类桶与训练参考类别错位")
    if not bins:
        return {"accepted": False, "per_bin": [], "overlapping_families": [],
                "min_distinct_train_children": min_distinct_children,
                "uses_validation_labels": False}
    vectors = _unit_rows(candidate_vectors)
    bin_vectors = _unit_rows(shared.text(bins))
    if vectors.shape[1] != bin_vectors.shape[1]:
        raise ValueError("分类桶与训练参考的 CLIP 嵌入维度不一致")
    example_labels = {_normalized(item.label) for item in references}
    forbidden = {"object", "objects", "thing", "things", "item", "items",
                 "background", "scene", "image", "photo", "stuff", "other"}
    per_bin = []
    support_sets: list[set[str]] = []
    all_accepted = True
    scores = vectors @ bin_vectors.T
    for column, name in enumerate(bins):
        normalized = _normalized(name)
        similarity = scores[:, column]
        peak = float(similarity.max())
        # 同一个训练子类仅归到最相近的一个桶，并要求领先第二桶；
        # 避免原先 .8*peak 宽阈值让每个上位桶都覆盖 300+ 个相同子类。
        competitor = np.max(np.delete(scores, column, axis=1), axis=1) if len(bins) > 1 else np.zeros(len(vectors))
        floor = 0.20
        ranked = np.argsort(-similarity, kind="stable")
        child_indices = []
        for index in ranked:
            if float(similarity[index]) < floor:
                break
            if float(similarity[index] - competitor[index]) < assignment_margin:
                continue
            # “cell phone”与“smartphone”不是两个不同的子类。
            if any(float(vectors[index] @ vectors[prior]) >= 0.84
                   for prior in child_indices):
                continue
            child_indices.append(int(index))
        support = {candidate_classes[index].label for index in child_indices}
        reasons = []
        if normalized in example_labels:
            reasons.append("repeats_single_reference_example")
        if normalized in forbidden:
            reasons.append("not_visually_locatable_object_family")
        if len(support) < min_distinct_children:
            reasons.append("too_few_distinct_train_child_classes")
        if reasons:
            all_accepted = False
        support_sets.append(support)
        per_bin.append({"bin": name, "accepted": not reasons,
                        "reasons": reasons, "peak_clip_cosine": peak,
                        "support_floor": floor,
                        "exclusive_assignment_margin": assignment_margin,
                        "distinct_train_child_count": len(support),
                        "support_examples": [candidate_classes[index].label
                                             for index in child_indices[:8]]})
    # 不同桶若主要解释同一批训练类，仍属于重复对象族。
    overlaps = []
    for left in range(len(bins)):
        for right in range(left + 1, len(bins)):
            union = support_sets[left] | support_sets[right]
            overlap = len(support_sets[left] & support_sets[right]) / len(union) if union else 0.0
            if overlap >= 0.6:
                overlaps.append({"left": bins[left], "right": bins[right],
                                 "support_jaccard": overlap})
                all_accepted = False
    return {"accepted": all_accepted, "per_bin": per_bin,
            "overlapping_families": overlaps,
            "min_distinct_train_children": min_distinct_children,
            "assignment_policy": "exclusive_best_bin_with_margin",
            "exclusive_assignment_margin": assignment_margin,
            "uses_validation_labels": False}


def reaudit_saved_qwen_outputs(
    source_audit: dict[str, Any], shared: Any, references: list[ReferenceClass],
    candidate_classes: list[ReferenceClass], candidate_vectors: np.ndarray,
) -> tuple[list[str] | None, list[dict[str, Any]]]:
    """仅重审已保存的真实 Qwen 原文，绝不新增或替换模型候选。"""
    if (source_audit.get("uses_validation_labels") is not False or
            not isinstance(source_audit.get("attempts"), list) or
            not source_audit["attempts"]):
        raise ValueError("旧审计缺少 train-only/Qwen 原文凭据")
    attempts: list[dict[str, Any]] = []
    for old in source_audit["attempts"]:
        if not isinstance(old, dict) or not isinstance(old.get("raw_model_output"), str):
            raise ValueError("旧审计的 Qwen 原文记录无效")
        raw = old["raw_model_output"]
        selection = deduplicate_semantic_bins(raw, shared)
        original_bins = list(selection["bins"])
        family = audit_object_families(
            original_bins, shared, references,
            candidate_classes, candidate_vectors)
        family_rounds = [{"bins": list(original_bins), "audit": family}]
        removed_unsupported: list[dict[str, Any]] = []
        # 只删除 Qwen 已提出但训练参考无法支撑的桶；每删一个重新做独占分配。
        # 其他失败原因（如抽象类别、复述 RAG 实例）不会自动删改。
        while not family["accepted"] and len(selection["bins"]) > 4:
            unsupported = [
                (index, row) for index, row in enumerate(family["per_bin"])
                if "too_few_distinct_train_child_classes" in row["reasons"]]
            if not unsupported:
                break
            index, row = min(unsupported, key=lambda item: (
                item[1]["distinct_train_child_count"], item[0]))
            removed_unsupported.append({
                "name": selection["bins"][index],
                "original_source_index": original_bins.index(selection["bins"][index]),
                "distinct_train_child_count_before_removal": row["distinct_train_child_count"],
                "required_distinct_train_children": family["min_distinct_train_children"],
                "reasons": list(row["reasons"]),
            })
            selection["bins"] = [name for position, name in enumerate(selection["bins"])
                                 if position != index]
            family = audit_object_families(
                selection["bins"], shared, references,
                candidate_classes, candidate_vectors)
            family_rounds.append({"bins": list(selection["bins"]), "audit": family})
        accepted = accepted_bin_selection(selection) and family["accepted"]
        # 每个 train 子类在 family audit 中至多支持一个桶；总数是可审计的覆盖量。
        coverage = sum(item["distinct_train_child_count"] for item in family["per_bin"])
        attempts.append({"source_turn": old.get("turn"), "raw_model_output": raw,
                         "raw_count": len(selection["raw_candidates"]),
                         "distinct_count": len(selection["bins"]),
                         "covered_distinct_train_children": coverage,
                         "original_bins_before_unsupported_filter": original_bins,
                         "removed_unsupported": removed_unsupported,
                         "family_audit_rounds": family_rounds,
                         "accepted": accepted, **selection,
                         "family_audit": family})
    selected_index = select_reaudited_attempt(attempts)
    return (attempts[selected_index]["bins"] if selected_index is not None else None), attempts


def select_reaudited_attempt(attempts: list[dict[str, Any]]) -> int | None:
    """优先更多合格上位族，再比 train 子类覆盖数，同分保留先出现的回答。"""
    best_index = None
    best_score = (-1, -1)
    for index, attempt in enumerate(attempts):
        if attempt.get("accepted") is not True:
            continue
        score = (int(attempt["distinct_count"]),
                 int(attempt["covered_distinct_train_children"]))
        if score > best_score:
            best_index, best_score = index, score
    return best_index
