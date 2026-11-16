"""按用户指定的 60 类 COCO + FSS 非测试 760 类准备首轮短训练。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml
from PIL import Image

from replay.core.config import load_config, write_json
from replay.data.manifest import _coco_rows, _fss_rows, _write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base = load_config(ROOT / "configs" / "image_autonomous_driving.yaml")
    def read_classes(key: str) -> set[str]:
        return {line.strip() for line in base.data_path(key).read_text(encoding="utf-8").splitlines()
                if line.strip()}

    fss_train = read_classes("fss_train_classes")
    fss_val = read_classes("fss_val_classes")
    fss_test = read_classes("fss_test_classes")
    if (len(fss_train), len(fss_val), len(fss_test)) != (520, 240, 240):
        raise ValueError("FSS 类别数量与 520/240/240 划分不符")
    if fss_train & fss_val or (fss_train | fss_val) & fss_test:
        raise ValueError("FSS 非测试类别与官方测试类别重叠")
    non_test = fss_train | fss_val
    if len(non_test) != 760:
        raise ValueError("FSS 非测试类别应恰好为 760")

    coco = base.data_path("coco2014_root")
    # 验证只用 COCO val2014 中属于当前 60 个基础类别的图；20 个留出类不作调参集。
    train_rows = list(_coco_rows(coco, "train", 0, category_role="train"))
    fss_rows = list(_fss_rows(base.data_path("fss1000_root"), "train", non_test))
    for row in fss_rows:
        with Image.open(row["image_path"]) as image:
            row["width"], row["height"] = image.size
    train_rows.extend(fss_rows)
    val_rows = list(_coco_rows(coco, "val", 0, category_role="train"))
    if len(fss_rows) != 7600:
        raise ValueError(f"FSS 非测试 760 类应有 7600 张图，实际 {len(fss_rows)}")
    if {row["image_path"] for row in train_rows} & {row["image_path"] for row in val_rows}:
        raise ValueError("训练和验证图像路径重叠")

    train_path = ROOT / "data/manifests/pilot_fold0_train.jsonl"
    val_path = ROOT / "data/manifests/pilot_fold0_val.jsonl"
    train_count = _write_jsonl(train_path, train_rows)
    val_count = _write_jsonl(val_path, val_rows)
    values = copy.deepcopy(base.values)
    values["data"].update({
        "train_manifest": str(train_path.relative_to(ROOT)),
        "val_manifest": str(val_path.relative_to(ROOT)),
        "kb_path": "artifacts/posterior_kb_pilot_fold0",
        "checkpoint": "artifacts/receiver_pilot_fold0.pt",
    })
    values["receiver"]["epochs"] = 1
    values["receiver"]["batch_size"] = 2
    config_path = ROOT / "configs/pilot_fold0_one_epoch.yaml"
    config_path.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {
        "fold": 0, "coco_train_images": train_count - len(fss_rows),
        "fss_non_test_train_images": len(fss_rows), "coco_val_images": val_count,
        "fss_official_test_classes_excluded": len(fss_test),
        "class_rule": "COCO-20^0: 60 base classes; FSS-1000: all 760 non-test classes",
        "validation_rule": "COCO val2014 images with at least one of the same 60 base classes",
        "training_epochs": 1, "precision": "float32", "resize_original_images": False,
        "per_gpu_batch_size_max": 2, "world_size": 3, "global_batch_size_max": 6,
        "batch_rule": "same-size groups; area <= 307200: up to 2/GPU, else 1/GPU",
        "config": str(config_path.relative_to(ROOT)),
    }
    write_json(ROOT / "artifacts/pilot_fold0_data_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
