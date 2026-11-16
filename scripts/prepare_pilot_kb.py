"""从已有第 0 折 KB 增量加入 FSS 非测试类别，另存短训练专用 KB。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from PIL import Image
import torch
from tqdm import tqdm

from replay.core.config import load_config, write_json
from replay.data.manifest import read_jsonl
from replay.knowledge.encoders import SharedKB
from replay.knowledge.store import PosteriorKB


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs/pilot_fold0_one_epoch.yaml")
    source = ROOT / "artifacts/posterior_kb"
    destination = config.data_path("kb_path")
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有知识库：{destination}")
    kb = PosteriorKB.load(source)
    existing = {record["id"] for record in kb.records}
    shared = SharedKB.from_config(config, torch.device("cuda"))
    text_cache = {}
    added = 0
    for row in tqdm(read_jsonl(config.data_path("train_manifest")), desc="FSS 760 KB 补充"):
        if row["source"] != "fss1000":
            continue
        name = row["bins"][0]
        record_id = f"fss1000:{row['source_id']}:{name}"
        if record_id in existing:
            continue
        descriptor = f"fss1000 image: {name}"
        if descriptor not in text_cache:
            text_cache[descriptor] = shared.text([descriptor])[0]
        with Image.open(row["image_path"]) as handle:
            data_vector = shared.images([handle.convert("RGB")])[0]
        kb.add(text_cache[descriptor], {
            "id": record_id, "split": "train", "source": "fss1000",
            "image_path": row["image_path"], "text": descriptor, "tags": [name],
        }, data_vector)
        added += 1
    if added != 2400:
        raise ValueError(f"预期补充 2400 张 FSS 原验证图，实际 {added}")
    kb.directory = destination
    kb.save()
    counts = Counter(record["source"] for record in kb.records)
    report = {"source": str(source), "destination": str(destination),
              "added_fss_non_test_records": added, "total_records": len(kb.records),
              "sources": dict(counts)}
    write_json(ROOT / "artifacts/pilot_fold0_kb_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
