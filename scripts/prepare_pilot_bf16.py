"""复制已核对的 COCO/FSS 第 0 折数据配置，生成独立的四卡 BF16 一轮实验。"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import yaml

from replay.core.config import load_config, write_json
from replay.data.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs 必须为正")
    previous = load_config(ROOT / "configs/pilot_fold0_one_epoch.yaml")
    train = read_jsonl(previous.data_path("train_manifest"))
    val = read_jsonl(previous.data_path("val_manifest"))
    train_sources = Counter(row["source"] for row in train)
    val_sources = Counter(row["source"] for row in val)
    expected_train = {"coco_exploratory": 74416, "fss1000": 7600}
    if dict(train_sources) != expected_train or dict(val_sources) != {"coco_exploratory": 36623}:
        raise ValueError("COCO/FSS 清单与上一轮已核对的数量不一致；不允许混入 Cityscapes")
    values = copy.deepcopy(previous.values)
    suffix = "one_epoch" if args.epochs == 1 else f"{args.epochs}_epochs"
    values["data"]["checkpoint"] = (
        "artifacts/receiver_pilot_fold0_bf16.pt" if args.epochs == 1
        else f"artifacts/receiver_pilot_fold0_bf16_{args.epochs}ep.pt")
    receiver = values["receiver"]
    receiver["epochs"] = args.epochs
    receiver["precision"] = "bfloat16"
    receiver["world_size"] = 4
    receiver["batch_size"] = 32  # 全局最大 128 仅用于 224×224 FSS；COCO 每卡最多 8。
    receiver["area_batch_rules"] = [
        {"max_pixels": 224 * 224, "batch_size": 32},
        {"max_pixels": 640 * 640, "batch_size": 8},
    ]
    destination = ROOT / f"configs/pilot_fold0_bf16_{suffix}.yaml"
    destination.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
    report = {
        "fold": 0, "train_sources": dict(train_sources), "val_sources": dict(val_sources),
        "train_images": len(train), "val_images": len(val),
        "cityscapes_in_training": False, "cityscapes_role": "later downstream evaluation only",
        "epochs": args.epochs, "world_size": 4,
        "precision": "BF16 autocast; FP32 weights/Adam/loss/token cache",
        "resize_original_images": False,
        "area_batch_rules": receiver["area_batch_rules"],
        "batch_selection": "benchmarked 224x224 and 640x480/640x640 on RTX 4090",
        "config": str(destination.relative_to(ROOT)),
    }
    audit = ("artifacts/pilot_fold0_bf16_data_audit.json" if args.epochs == 1
             else f"artifacts/pilot_fold0_bf16_{args.epochs}ep_data_audit.json")
    write_json(ROOT / audit, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
