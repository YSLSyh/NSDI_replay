"""用 Qwen4B/SharedKB 和多样 train-only KB 参考生成图 7 分类语义桶。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tx_stage12_qwen4b.yaml"))
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--kb-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="使用新路径；已有输出保留作诊断，不覆盖")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-classes", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    if args.task_description.strip().lower() == "autonomous driving":
        parser.error("图 7 分类语义桶不能复用自动驾驶任务描述")
    if args.context_classes < 8 or not 1 <= args.max_attempts <= 6:
        parser.error("至少需要 8 个多样参考类，重试次数须在 1–6")

    import torch
    from replay.core.config import write_json
    from replay.evaluation.figure7_bins import (
        CLASSIFICATION_SYSTEM_PROMPT, accepted_bin_selection, audit_object_families,
        deduplicate_semantic_bins, select_diverse_references,
        unique_train_reference_classes,
    )
    from replay.evaluation.figure7_kdc import Figure7KdcSpec, _classification_config
    from replay.knowledge.bins import _load_qwen, _qwen_answer
    from replay.knowledge.encoders import SharedKB
    from replay.knowledge.store import PosteriorKB

    spec = Figure7KdcSpec(
        config_path=args.config, task_description=args.task_description,
        bins_cache=args.output, kb_path=args.kb_path,
        receiver_checkpoint=Path("unused.pt"),
        fixed_loss_budget_bytes=1)
    config = _classification_config(spec)
    destination = config.path(args.output)
    if destination.exists():
        raise FileExistsError(f"语义桶缓存已存在，拒绝覆盖：{destination}")
    kb = PosteriorKB.load(config.path(args.kb_path))
    candidates = unique_train_reference_classes(kb.records)
    device = torch.device(args.device)
    shared = SharedKB.from_config(config, device)

    # CLIP 文本编码复用现有 SharedKB；每个 (来源, 标签) 仅编码一次。
    label_vectors = np.concatenate([
        shared.text([item.label for item in candidates[start:start + 128]])
        for start in range(0, len(candidates), 128)
    ])
    task_vector = shared.text([args.task_description])[0]
    references = select_diverse_references(
        candidates, label_vectors, task_vector, count=args.context_classes)
    source_counts = Counter(item.source for item in references)
    context = "\n".join(
        f"- {item.label} (train reference: {item.source})" for item in references)
    prompt = (
        f'Task scenario: "{args.task_description}"\n'
        "Training references below are examples of the data distribution, NOT answer choices.\n"
        f"{context}\n"
        "This task must recognize a broad range of everyday objects and living things, "
        "not just the individual examples listed above. Generate 4–8 HIGH-LEVEL, "
        "VISUALLY LOCALIZABLE OBJECT FAMILIES suitable as CLIPSeg region prompts. "
        "Each family must cover several visibly different object classes in the "
        "training references; it must not be a single object class, a synonym, "
        "a subtype, or a vague abstract category such as 'things'. "
        "Cover different parts of the broad task, including living and manufactured "
        "objects when the references support them. "
        "Do not copy any single reference label as an answer and do not infer "
        "anything from validation labels. Return only comma-separated family names.\n"
        "Object families:")
    tokenizer, model = _load_qwen(config, device)
    messages = [{"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    attempts = []
    chosen = None
    for turn in range(args.max_attempts):
        answer = _qwen_answer(tokenizer, model, device, messages)
        selection = deduplicate_semantic_bins(answer, shared)
        family = audit_object_families(
            selection["bins"], shared, references, candidates, label_vectors)
        raw_count = len(selection["raw_candidates"])
        distinct_count = len(selection["bins"])
        accepted = accepted_bin_selection(selection) and family["accepted"]
        attempts.append({"turn": turn + 1, "raw_model_output": answer,
                         "raw_count": raw_count, "distinct_count": distinct_count,
                         "accepted": accepted, **selection,
                         "family_audit": family})
        if accepted:
            chosen = selection["bins"]
            break
        if turn + 1 == args.max_attempts:
            break
        duplicates = ", ".join(
            f"{item['name']}≈{item['same_as']}" for item in selection["removed_synonyms"])
        family_failures = "; ".join(
            f"{item['bin']}: {', '.join(item['reasons'])}"
            for item in family["per_bin"] if not item["accepted"])
        messages.extend([
            {"role": "assistant", "content": answer},
            {"role": "user", "content": (
                f"Your answer had {raw_count} names but only {distinct_count} distinct "
                f"semantic classes after synonym/subtype filtering. "
                f"Near duplicates: {duplicates or 'none'}. "
                f"Family audit rejected: {family_failures or 'none'}. "
                "Return 4–8 visually localizable PARENT object families. "
                "Each must cover at least three DIFFERENT training object classes, "
                "and must not copy a listed reference example or name a single "
                "object type. Choose unrelated families across the broad task. "
                "Return only comma-separated family names."
            )},
        ])

    audit = {
        "task": args.task_description,
        "kb_path": str(config.path(args.kb_path)),
        "kb_record_count": len(kb.records),
        "unique_train_reference_classes": len(candidates),
        "balanced_context_classes": [
            {"source": item.source, "label": item.label, "record_id": item.record_id}
            for item in references],
        "balanced_context_source_counts": dict(source_counts),
        "reference_selection": "source round-robin + CLIP text novelty MMR",
        "family_validation": "exclusive train class assignment with margin and context-example exclusion",
        "semantic_duplicate_threshold": 0.94,
        "attempts": attempts,
        "qwen_revision": config.section("models")["qwen"]["revision"],
        "clip_revision": config.section("models")["clip"]["revision"],
        "uses_validation_labels": False,
    }
    if chosen is None:
        diagnostic = destination.with_suffix(".failed.json")
        write_json(diagnostic, {**audit, "status": "rejected_insufficient_diversity"})
        raise ValueError(f"Qwen 未生成 4–8 个不同语义类；诊断已保存：{diagnostic}")
    result = {
        "task": args.task_description, "bins": chosen,
        "semantic_distinct_count": len(chosen),
        "semantic_diversity_verified": True,
        "broad_family_verified": True,
        "raw_model_output": attempts[-1]["raw_model_output"],
        "raw_model_outputs": [item["raw_model_output"] for item in attempts],
        "selection_audit": audit,
        "kb_reference_sources": dict(Counter(
            str(row.get("source", "unknown")) for row in kb.records
            if row.get("has_data") is True)),
        "qwen_revision": audit["qwen_revision"],
        "clip_revision": audit["clip_revision"],
    }
    write_json(destination, result)
    print(json.dumps({"output": str(destination), "task": result["task"],
                      "bins": chosen,
                      "balanced_context_source_counts": dict(source_counts),
                      "synonyms_removed": attempts[-1]["removed_synonyms"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
