"""从训练集构建可检索的 PosteriorKB；验证集和测试集绝不写入。"""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from replay.core.config import ReplayConfig
from replay.core.config import write_json
from replay.data.cityscapes import load_cityscapes_pair, TRAIN_NAMES
from replay.data.manifest import read_jsonl
from replay.knowledge.encoders import SharedKB
from replay.knowledge.store import PosteriorKB


def add_cityscapes_references(config: ReplayConfig, shared: SharedKB, kb: PosteriorKB) -> int:
    """将 Cityscapes train 的已标注区域增量写入 KB，不接触 val。"""
    initial = len(kb.records)
    # 有标注的 Cityscapes 训练图：每个出现的类别取可见区域的外接矩形作为参考。
    for row in tqdm(read_jsonl(config.data_path("reference_manifest")), desc="Cityscapes KB"):
        if row["split"] != "train":
            raise ValueError("知识库清单包含非训练数据")
        image, mask = load_cityscapes_pair(Path(row["image_path"]), Path(row["label_path"]))
        for train_id in np.unique(mask):
            if train_id == 255:
                continue
            ys, xs = np.where(mask == train_id)
            if len(xs) == 0:
                continue
            name = TRAIN_NAMES[int(train_id)]
            left, top = int(xs.min()), int(ys.min())
            right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
            # 极小标注框会让 CLIP 图像预处理器在单像素轴上丢失通道；只扩展取样窗口，不改原图。
            right = min(image.width, max(right, left + 16))
            bottom = min(image.height, max(bottom, top + 16))
            left = max(0, min(left, right - 16))
            top = max(0, min(top, bottom - 16))
            crop = image.crop((left, top, right, bottom))
            text = f"autonomous driving scene: {name}"
            kb.add(shared.text([text])[0],
                   {"id": f"cityscapes:{row['source_id']}:{name}", "split": "train",
                    "source": "cityscapes", "image_path": row["image_path"],
                    "text": text, "tags": [name]},
                   shared.images([crop])[0])
    return len(kb.records) - initial


def extend_cityscapes_kb(config: ReplayConfig, shared: SharedKB) -> PosteriorKB:
    """保留正在训练使用的预训练 KB，另存加入 Cityscapes 的完整版本。"""
    source = config.data_path("kb_path")
    destination = config.data_path("kb_full_path")
    if destination.exists():
        raise FileExistsError(f"完整 KB 目标已存在，拒绝覆盖：{destination}")
    kb = PosteriorKB.load(source)
    kb.directory = destination
    added = add_cityscapes_references(config, shared, kb)
    if added == 0:
        raise ValueError("Cityscapes 参考清单未产生任何区域特征")
    kb.save()
    return kb


def build_kb(config: ReplayConfig, shared: SharedKB, pretrain_only: bool = False) -> PosteriorKB:
    kb = PosteriorKB(config.data_path("kb_path"))
    taxonomy = json.loads(config.path(config.section("task")["taxonomy"]).read_text(encoding="utf-8"))
    for item in taxonomy["entries"]:
        text = f"{item['name']}: {item['description']}"
        kb.add(shared.text([text])[0], {"id": f"taxonomy:{item['name']}", "split": "train",
                                           "source": "taxonomy", "text": text, "tags": [item["name"]]})
    if not pretrain_only:
        add_cityscapes_references(config, shared, kb)

    # COCO/FSS 恢复训练样本也提供图像参考；其类别来自已有数据集标注，无须新人工标注。
    for row in tqdm(read_jsonl(config.data_path("train_manifest")), desc="Restoration KB"):
        if row["split"] != "train":
            raise ValueError("知识库清单包含非训练数据")
        with Image.open(row["image_path"]) as handle:
            image_vector = shared.images([handle.convert("RGB")])[0]
        for name in row["bins"]:
            text = f"{row['source']} image: {name}"
            kb.add(shared.text([text])[0],
                   {"id": f"{row['source']}:{row['source_id']}:{name}", "split": "train",
                    "source": row["source"], "image_path": row["image_path"],
                    "text": text, "tags": [name]}, image_vector)
    kb.save()
    return kb


def cityscapes_only_inputs(config: ReplayConfig, destination: Path) -> tuple[list[dict], list[dict], dict]:
    """先核验 Cityscapes train 清单及 19 类词表，禁止混入其他来源。"""
    expected = (config.root / "artifacts/posterior_kb_cityscapes_train").resolve()
    if not expected.parent.is_relative_to(config.root.resolve()):
        raise ValueError("artifacts 路径越出 replay 项目，拒绝写入")
    if destination.resolve() != expected:
        raise ValueError(f"Cityscapes 专用 KB 只能写入 {expected}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有知识库：{destination}")

    manifest = config.data_path("reference_manifest")
    required_manifest = config.root / "data/manifests/cityscapes_reference_train.jsonl"
    if manifest.resolve() != required_manifest.resolve():
        raise ValueError("专用 KB 只接受 cityscapes_reference_train.jsonl 清单")
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("Cityscapes train 参考清单为空")

    taxonomy_path = config.path(config.section("task")["taxonomy"])
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    entries = taxonomy.get("entries")
    if not isinstance(entries, list) or [item.get("name") for item in entries] != TRAIN_NAMES:
        raise ValueError("Cityscapes taxonomy 必须按官方 trainId 顺序包含全部 19 类")
    if any(not isinstance(item.get("description"), str) or not item["description"].strip()
           for item in entries):
        raise ValueError("19 类 taxonomy 的说明文字不能为空")

    root = config.data_path("cityscapes_root").resolve()
    image_root = (root / "leftImg8bit/train").resolve()
    label_root = (root / "gtFine/train").resolve()
    seen: set[str] = set()
    verified_rows: list[dict] = []
    for row in rows:
        if row.get("source") != "cityscapes" or row.get("split") != "train":
            raise ValueError("参考清单包含非 Cityscapes train 的记录")
        image = Path(row["image_path"]).resolve(strict=True)
        label = Path(row["label_path"]).resolve(strict=True)
        # resolve 后再检查，防止符号链接把 val、tune 或其他数据带入知识库。
        try:
            image_relative = image.relative_to(image_root)
            label_relative = label.relative_to(label_root)
        except ValueError as error:
            raise ValueError("参考图或标注不在 Cityscapes train 目录内") from error
        if len(image_relative.parts) != 2 or len(label_relative.parts) != 2:
            raise ValueError("Cityscapes 图像和标注必须位于城市子目录")
        if not image.name.endswith("_leftImg8bit.png"):
            raise ValueError(f"非法 Cityscapes 图像文件名：{image}")
        expected_label = image.name.removesuffix("_leftImg8bit.png") + "_gtFine_labelIds.png"
        if label.parent.name != image.parent.name or label.name != expected_label:
            raise ValueError(f"Cityscapes 图像与标注不配对：{image}")
        if row.get("source_id") != image.stem or image.stem in seen:
            raise ValueError(f"Cityscapes source_id 缺失、错误或重复：{image}")
        seen.add(image.stem)
        verified_rows.append({"image_path": str(image), "label_path": str(label),
                              "source_id": image.stem})

    source_audit = {
        "reference_manifest": str(manifest.resolve()),
        "reference_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "taxonomy": str(taxonomy_path.resolve()),
        "taxonomy_sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
        "reference_images": len(verified_rows),
        "allowed_sources": ["taxonomy", "cityscapes"],
        "allowed_split": "train",
    }
    return verified_rows, entries, source_audit


def build_cityscapes_only_kb(config: ReplayConfig, shared: SharedKB, destination: Path,
                             rows: list[dict], entries: list[dict], source_audit: dict) -> dict:
    """只用 Cityscapes train 标注区域建立独立 KB，完整写入后才发布目标目录。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有知识库：{destination}")
    staging = Path(tempfile.mkdtemp(prefix=".posterior_kb_cityscapes_train.",
                                    dir=destination.parent))
    try:
        kb = PosteriorKB(staging)
        taxonomy_texts = [f"{item['name']}: {item['description']}" for item in entries]
        class_texts = [f"autonomous driving scene: {name}" for name in TRAIN_NAMES]
        # 文本只编码两批，避免对几千张图的每个类别重复运行 CLIP 文本塔。
        taxonomy_vectors = shared.text(taxonomy_texts)
        class_vectors = shared.text(class_texts)
        for name, descriptor, vector in zip(TRAIN_NAMES, taxonomy_texts, taxonomy_vectors):
            kb.add(vector, {"id": f"taxonomy:{name}", "split": "train", "source": "taxonomy",
                            "text": descriptor, "tags": [name]})

        class_counts: Counter[str] = Counter()
        for row in tqdm(rows, desc="Cityscapes-only KB"):
            image, mask = load_cityscapes_pair(Path(row["image_path"]), Path(row["label_path"]))
            names: list[str] = []
            boxes: list[tuple[int, int, int, int]] = []
            crops: list[Image.Image] = []
            for train_id in np.unique(mask):
                if train_id == 255:
                    continue
                ys, xs = np.where(mask == train_id)
                left, top = int(xs.min()), int(ys.min())
                right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
                # 只扩展极小标注框的特征提取窗口；原图和标注均不改动。
                right = min(image.width, max(right, left + 16))
                bottom = min(image.height, max(bottom, top + 16))
                left = max(0, min(left, right - 16))
                top = max(0, min(top, bottom - 16))
                name = TRAIN_NAMES[int(train_id)]
                names.append(name)
                boxes.append((left, top, right, bottom))
                crops.append(image.crop(boxes[-1]))
            # 每张图的所有标注区域一次批量编码，减少 GPU 调用与预处理开销。
            if crops:
                vectors = shared.images(crops)
                for name, box, vector in zip(names, boxes, vectors):
                    descriptor = class_texts[TRAIN_NAMES.index(name)]
                    kb.add(class_vectors[TRAIN_NAMES.index(name)], {
                        "id": f"cityscapes:{row['source_id']}:{name}",
                        "split": "train", "source": "cityscapes",
                        "image_path": row["image_path"], "label_path": row["label_path"],
                        "bbox_xyxy": list(box), "text": descriptor, "tags": [name],
                    }, vector)
                    class_counts[name] += 1
            for crop in crops:
                crop.close()
            image.close()

        region_count = sum(class_counts.values())
        if region_count == 0:
            raise ValueError("Cityscapes train 标注未产生任何类别参考图")
        sources = dict(Counter(record["source"] for record in kb.records))
        if sources != {"taxonomy": 19, "cityscapes": region_count}:
            raise ValueError(f"知识库混入了不允许的来源：{sources}")
        audit = {**source_audit, "destination": str(destination),
                 "source_counts": sources, "region_references": region_count,
                 "total_records": len(kb.records),
                 "class_reference_counts": {name: class_counts[name] for name in TRAIN_NAMES},
                 "covered_classes": [name for name in TRAIN_NAMES if class_counts[name] > 0],
                 "uncovered_classes": [name for name in TRAIN_NAMES if class_counts[name] == 0]}
        kb.save()
        write_json(staging / "build_audit.json", audit)
        persisted = PosteriorKB.load(staging)
        if len(persisted.records) != audit["total_records"] or sum(
                record["source"] == "cityscapes" and record["has_data"]
                for record in persisted.records) != region_count:
            raise ValueError("Cityscapes 专用 KB 落盘后记录或图像特征数量不一致")
        # 同一 artifacts 文件系统内重命名；失败时只清理本次创建的临时目录。
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"目标知识库在构建过程中已出现：{destination}")
        staging.rename(destination)
        return audit
    finally:
        if staging.exists():
            if staging.resolve().parent != destination.parent.resolve():
                raise RuntimeError("临时知识库目录越界，拒绝清理")
            shutil.rmtree(staging)
