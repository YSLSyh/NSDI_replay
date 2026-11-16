"""论文图 7 的 ImageNet-1K Top-1 评测核心与结果审计。

这里仅定义完整验证集、分类器及统一的逐图结果格式。论文没有公开分类器
型号；任何选用的权重必须记录为 replay 的实现选择。缺少官方验证集或任一
方法的完整逐图结果时，禁止绘制标为论文图 7 复现的曲线。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image


LOSS_METHODS = ("prog_jpeg", "limitnet", "starfish", "kdc")
BUDGET_METHODS = ("jpeg", "prog_jpeg", "limitnet", "starfish", "kdc")
EXPECTED_CLASSES = 1_000
EXPECTED_IMAGES = 50_000
EXPECTED_PER_CLASS = 50
MIN_FORMAL_INTACT_TOP1 = 0.84
_WNID = re.compile(r"n\d{8}\Z")
_JPEG_SUFFIXES = {".jpeg", ".jpg"}
_OFFICIAL_VAL_FILENAME = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG\Z")


@dataclass(frozen=True)
class ValSample:
    """类别索引按 ImageFolder/torchvision 标准的 WNID 字典序生成。"""

    image_id: str
    path: Path
    target: int
    wnid: str


def validate_imagenet_val(
    root: Path, *, expected_classes: int = EXPECTED_CLASSES,
    expected_images: int = EXPECTED_IMAGES,
    expected_per_class: int = EXPECTED_PER_CLASS,
) -> tuple[list[ValSample], str]:
    """要求 val/<synset WNID>/*.JPEG 完整且每类数量一致。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet-1K val 不存在：{root}")
    class_dirs = sorted(child for child in root.iterdir() if child.is_dir())
    if (len(class_dirs) != expected_classes or
            any(not _WNID.fullmatch(folder.name) for folder in class_dirs)):
        raise ValueError(f"ImageNet val 应有 {expected_classes} 个 WNID 类别目录")
    if any(child.is_file() for child in root.iterdir()):
        raise ValueError("验证图必须按 WNID 类别目录组织，避免标签顺序错位")
    samples: list[ValSample] = []
    official_indices: set[int] = set()
    digest = sha256()
    for class_index, folder in enumerate(class_dirs):
        files = sorted(file for file in folder.iterdir()
                       if file.is_file() and file.suffix.lower() in _JPEG_SUFFIXES)
        if len(files) != expected_per_class or any(child.is_dir() for child in folder.iterdir()):
            raise ValueError(f"{folder.name} 应有恰好 {expected_per_class} 张 JPEG 验证图")
        for file in files:
            if expected_images == EXPECTED_IMAGES:
                match = _OFFICIAL_VAL_FILENAME.fullmatch(file.name)
                if match is None or not 1 <= int(match.group(1)) <= EXPECTED_IMAGES:
                    raise ValueError(f"文件名不符合官方 ILSVRC2012 val 格式：{file.name}")
                if int(match.group(1)) in official_indices:
                    raise ValueError(f"验证图编号重复：{file.name}")
                official_indices.add(int(match.group(1)))
            # 类别目录和文件名共同构成稳定 ID；文件大小可发现数据内容更换。
            image_id = f"{folder.name}/{file.name}"
            digest.update(f"{image_id}\t{class_index}\t{file.stat().st_size}\n".encode())
            samples.append(ValSample(image_id, file, class_index, folder.name))
    if (len(samples) != expected_images or
            len({item.image_id for item in samples}) != len(samples) or
            (expected_images == EXPECTED_IMAGES and len(official_indices) != EXPECTED_IMAGES)):
        raise ValueError(f"ImageNet val 应有 {expected_images} 张互不重复的图像")
    return samples, digest.hexdigest()


def _weight_sha256(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


class Classifier:
    """只做下游分类；编解码始终作用于输入原图。"""

    def __init__(self, *, backend: str, weights_path: Path, device: str):
        import torch

        self.backend = backend
        self.device = torch.device(device)
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.weights_path = path
        self.weights_sha256 = _weight_sha256(path)
        if (backend == "torchvision_vit_l_16_swag_e2e" and
                not self.weights_sha256.startswith("4f3808c9")):
            raise ValueError("ViT-L/16 权重不是指定的官方 SWAG E2E 版本")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if backend == "efficientnet_b0":
            from efficientnet_pytorch import EfficientNet

            self.model = EfficientNet.from_name("efficientnet-b0", num_classes=1_000)
            self.model.load_state_dict(state, strict=True)
            self.transforms = None
        elif backend == "torchvision_vit_l_16_swag_e2e":
            from torchvision.models import ViT_L_16_Weights, vit_l_16

            # SWAG E2E 权重使用 512×512，位置编码长度与默认 224 模型不同。
            self.model = vit_l_16(weights=None, image_size=512)
            self.model.load_state_dict(state, strict=True)
            self.transforms = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        else:
            raise ValueError(f"未支持的分类器：{backend}")
        self.model = self.model.to(self.device).eval()

    def _preprocess(self, image: Image.Image):
        import torch

        rgb = image.convert("RGB")
        if self.transforms is not None:
            return self.transforms(rgb).unsqueeze(0).to(self.device)
        # EfficientNet-PyTorch 官方 B0 输入为 224 中心裁块和 ImageNet 标准归一化。
        width, height = rgb.size
        scale = 256 / min(width, height)
        resized = rgb.resize((round(width * scale), round(height * scale)),
                             Image.Resampling.BICUBIC)
        left = (resized.width - 224) // 2
        top = (resized.height - 224) // 2
        crop = resized.crop((left, top, left + 224, top + 224))
        array = np.asarray(crop, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> int:
        import torch

        with torch.inference_mode():
            logits = self.model(self._preprocess(image))
        if logits.shape != (1, 1_000):
            raise ValueError("分类器输出不是 1000 类 logits")
        return int(logits.argmax(dim=1).item())

    def predict_batch(self, images: list[Image.Image]) -> list[int]:
        """逐图使用同一预处理，合并前向计算以加速原图 Top-1 验证。"""
        if not images:
            raise ValueError("分类批次不能为空")
        if len(images) == 1:
            return [self.predict(images[0])]
        import torch

        # _preprocess 与单图评测完全相同；这里只合并模型前向，避免改变输入。
        with torch.inference_mode():
            batch = torch.cat([self._preprocess(image) for image in images], dim=0)
            logits = self.model(batch)
        if logits.shape != (len(images), 1_000):
            raise ValueError("分类器输出不是每张图 1000 类 logits")
        return [int(value) for value in logits.argmax(dim=1).cpu().tolist()]


class ClassificationCodec(Protocol):
    """后续 KDC/基线适配器的统一契约。"""

    name: str

    def encode(self, image: Image.Image, *, budget_bytes: int | None) -> tuple[bytes, dict]: ...

    def decode(self, payload: bytes, *, loss_flags: tuple[bool, ...],
               received_prefix_bytes: int | None) -> Image.Image | None: ...


def evaluate_codec_point(
    *, samples: list[ValSample], classifier: Classifier,
    codecs: Mapping[str, ClassificationCodec], axis: str, x: float | int,
    loss_rate: float, packet_payload_bytes: int, channel_seed: int,
) -> list[dict[str, Any]]:
    """对所有方法使用相同图像、标签和逐包随机数序列。返回逐图记录。

    每种码流长度不同，实际丢包数可不同；同一图像/点位的第 i 个包始终
    使用同一个随机数。此函数只运行给定样本，正式报告另要求完整 50k。
    """
    if axis not in ("loss", "budget") or not 0 <= loss_rate <= 1:
        raise ValueError("横轴或丢包率无效")
    if packet_payload_bytes <= 0 or not codecs:
        raise ValueError("分包字节数或方法集合无效")
    point = _point_key(axis, x)
    budget = int(point) if axis == "budget" else None
    rows: list[dict[str, Any]] = []
    for sample in samples:
        token = f"{channel_seed}\t{axis}\t{point}\t{sample.image_id}"
        event_id = sha256(token.encode()).hexdigest()
        event_seed = int(event_id[:16], 16)
        with Image.open(sample.path) as source:
            image = source.convert("RGB")
        for method, codec in codecs.items():
            if codec.name != method:
                raise ValueError(f"方法名称与编解码器不一致：{method}")
            payload, metadata = codec.encode(image, budget_bytes=budget)
            sent = metadata.get("wire_bytes")
            if type(sent) is not int or sent < len(payload):
                raise ValueError(f"{method} 未报告有效的实际传输字节数")
            if budget is not None and sent > budget:
                raise ValueError(f"{method} 实际传输字节数超过预算")
            jpeg_fields: dict[str, int | None] = {}
            if method in ("jpeg", "prog_jpeg"):
                codec_bytes = metadata.get("codec_bytes")
                quality = metadata.get("jpeg_quality")
                if (type(codec_bytes) is not int or codec_bytes != len(payload) or
                        (quality is None and codec_bytes != 0) or
                        (quality is not None and
                         (type(quality) is not int or not 1 <= quality <= 100))):
                    raise ValueError(f"{method} 缺少有效的 JPEG 质量或码流字节数")
                jpeg_fields = {"jpeg_quality": quality, "codec_bytes": codec_bytes,
                               "wire_bytes": sent}
            packet_count = metadata.get(
                "loss_packet_count", math.ceil(len(payload) / packet_payload_bytes))
            if type(packet_count) is not int or packet_count < 0:
                raise ValueError(f"{method} 未报告有效的数据分包数量")
            generator = random.Random(event_seed)
            loss_flags = tuple(generator.random() < loss_rate
                               for _ in range(packet_count))
            decoded = codec.decode(payload, loss_flags=loss_flags,
                                   received_prefix_bytes=None)
            prediction = -1 if decoded is None else classifier.predict(decoded)
            rows.append({
                "axis": axis, "x": float(point) if axis == "loss" else budget,
                "method": method, "image_id": sample.image_id,
                "target": sample.target, "prediction": prediction,
                "sent_bytes": sent, "source_codec_bytes":
                    metadata.get("source_codec_bytes", len(payload)),
                "budget_policy": metadata.get("budget_policy"),
                "receiver_session_update": metadata.get("receiver_session_update"),
                "packet_count": packet_count, "lost_packets": sum(loss_flags),
                "loss_flags_sha256": sha256(bytes(loss_flags)).hexdigest(),
                "channel_event_id": event_id,
                **jpeg_fields,
            })
    return rows


def validate_intact_predictions(
    path: Path, samples: list[ValSample], *, minimum_top1: float = MIN_FORMAL_INTACT_TOP1,
) -> dict[str, Any]:
    """同一分类器先在无损原图上验收；避免错误标签和弱模型扭曲曲线。"""
    expected = {sample.image_id: sample.target for sample in samples}
    predictions: dict[str, int] = {}
    backend: str | None = None
    weights_sha256: str | None = None
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = row["image_id"]
            if (image_id not in expected or image_id in predictions or
                    row.get("target") != expected[image_id] or
                    type(row.get("prediction")) is not int or
                    not 0 <= row["prediction"] < EXPECTED_CLASSES or
                    row.get("backend") not in
                    ("torchvision_vit_l_16_swag_e2e", "efficientnet_b0") or
                    not re.fullmatch(r"[0-9a-f]{64}",
                                     str(row.get("weights_sha256", "")))):
                raise ValueError("完整图分类结果存在重复、标签错位或无效预测")
            if backend is None:
                backend = row["backend"]
                weights_sha256 = row["weights_sha256"]
            elif row["backend"] != backend or row["weights_sha256"] != weights_sha256:
                raise ValueError("完整图分类分片混用了不同分类器或权重")
            predictions[image_id] = row["prediction"]
    if len(predictions) != len(samples):
        raise ValueError("完整图分类结果未覆盖全部 ImageNet-1K 验证图")
    hits = sum(predictions[sample.image_id] == sample.target for sample in samples)
    top1 = hits / len(samples)
    if top1 < minimum_top1:
        raise ValueError(
            f"完整图 Top-1={top1:.4f} 低于 {minimum_top1:.2f} 门槛；"
            "检查模型、512×512 预处理和类别索引")
    low, high = wilson_95(hits, len(samples))
    return {"top1": top1, "hits": hits, "n": len(samples), "ci95": [low, high],
            "minimum_top1": minimum_top1, "backend": backend,
            "weights_sha256": weights_sha256}


def wilson_95(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Top-1 计数无效")
    z = 1.959963984540054
    p = successes / trials
    z2 = z * z
    center = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    radius = z * math.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / (1 + z2 / trials)
    return max(0.0, center - radius), min(1.0, center + radius)


def _point_key(axis: str, value: object) -> str:
    if axis == "loss":
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("丢包率必须在 [0,1]")
        return format(number, ".8g")
    if axis == "budget":
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError("传输预算必须是正整数字节")
        return str(int(value))
    raise ValueError("图 7 横轴必须是 loss 或 budget")


def summarize_figure7(
    *, val_root: Path, settings_path: Path, trials_path: Path, output_dir: Path,
    require_full_methods: bool = False,
) -> dict[str, Any]:
    """逐图结果不完整、标签不符或方法缺失时不生成论文式曲线。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    samples, dataset_sha = validate_imagenet_val(val_root)
    expected_targets = {sample.image_id: sample.target for sample in samples}
    settings_path = Path(settings_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if (settings.get("dataset_image_count") != EXPECTED_IMAGES or
            settings.get("dataset_manifest_sha256") != dataset_sha or
            settings.get("classification_metric") != "top1" or
            settings.get("split") != "val" or
            settings.get("classifier_is_paper_specified") is not False):
        raise ValueError("图 7 设置与完整 ImageNet-1K val 或模型来源不一致")
    classifier = settings.get("classifier")
    if (not isinstance(classifier, dict) or
            classifier.get("backend") != "torchvision_vit_l_16_swag_e2e" or
            not re.fullmatch(r"[0-9a-f]{64}", str(classifier.get("weights_sha256", "")))):
        raise ValueError("正式图 7 必须使用可达到论文完整图精度的分类器和已验签权重")
    intact_path = settings.get("intact_predictions_path")
    if not isinstance(intact_path, str) or not intact_path:
        raise ValueError("缺少完整图 Top-1 验收结果")
    intact_file = Path(intact_path)
    if not intact_file.is_absolute():
        intact_file = settings_path.parent / intact_file
    intact = validate_intact_predictions(intact_file, samples)
    if (intact["backend"] != classifier["backend"] or
            intact["weights_sha256"] != classifier["weights_sha256"]):
        raise ValueError("原图 Top-1 验收所用权重与图 7 设置不一致")
    if not isinstance(settings.get("channel"), dict) or not settings["channel"]:
        raise ValueError("图 7 必须记录信道和分包配置")
    expected_points: dict[str, set[str]] = {
        "loss": {_point_key("loss", value) for value in settings["loss_rates"]},
        "budget": {_point_key("budget", value) for value in settings["budgets_bytes"]},
    }
    expected_methods = {
        "loss": LOSS_METHODS if require_full_methods else tuple(settings["loss_methods"]),
        "budget": BUDGET_METHODS if require_full_methods else tuple(settings["budget_methods"]),
    }
    if (not expected_points["loss"] or not expected_points["budget"] or
            any(not methods or len(methods) != len(set(methods)) or
                not set(methods).issubset(set(LOSS_METHODS if axis == "loss"
                                              else BUDGET_METHODS))
                for axis, methods in expected_methods.items()) or
            (require_full_methods and
             (tuple(settings.get("loss_methods", ())) != LOSS_METHODS or
              tuple(settings.get("budget_methods", ())) != BUDGET_METHODS))):
        raise ValueError("图 7 方法或横轴配置不完整")
    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    correct: dict[tuple[str, str, str], int] = defaultdict(int)
    sent_sizes: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    jpeg_qualities: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    jpeg_codec_sizes: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    jpeg_quality_missing: dict[tuple[str, str, str], int] = defaultdict(int)
    event_ids: dict[tuple[str, str, str], str] = {}
    with Path(trials_path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            axis = row["axis"]
            x = _point_key(axis, row["x"])
            method = row["method"]
            image_id = row["image_id"]
            key = axis, x, method
            if (x not in expected_points[axis] or method not in expected_methods[axis] or
                    image_id not in expected_targets or
                    row["target"] != expected_targets[image_id] or
                    image_id in seen[key] or
                    type(row["prediction"]) is not int or
                    not -1 <= row["prediction"] < EXPECTED_CLASSES or
                    type(row["sent_bytes"]) is not int or row["sent_bytes"] < 0 or
                    (axis == "budget" and row["sent_bytes"] > int(x)) or
                    (method == "kdc" and row.get("receiver_session_update") is not False) or
                    not isinstance(row.get("channel_event_id"), str) or
                    not row["channel_event_id"]):
                raise ValueError("逐图 Top-1 结果重复、标签错误、超预算或方法不符")
            if method in ("jpeg", "prog_jpeg"):
                quality = row.get("jpeg_quality")
                codec_bytes = row.get("codec_bytes")
                wire_bytes = row.get("wire_bytes")
                if (type(codec_bytes) is not int or codec_bytes < 0 or
                        type(wire_bytes) is not int or wire_bytes != row["sent_bytes"] or
                        codec_bytes > wire_bytes or
                        (quality is None and (codec_bytes != 0 or wire_bytes != 0)) or
                        (quality is not None and
                         (type(quality) is not int or not 1 <= quality <= 100))):
                    raise ValueError("JPEG 逐图质量或码流/线字节记录无效")
                jpeg_codec_sizes[key].append(codec_bytes)
                if quality is None:
                    jpeg_quality_missing[key] += 1
                else:
                    jpeg_qualities[key].append(quality)
            event_key = axis, x, image_id
            prior_event = event_ids.setdefault(event_key, row["channel_event_id"])
            if prior_event != row["channel_event_id"]:
                raise ValueError("同一图像和横轴点的不同方法未使用配对信道事件")
            seen[key].add(image_id)
            correct[key] += int(row["prediction"] == row["target"])
            sent_sizes[key].append(row["sent_bytes"])
    report: dict[str, Any] = {
        "protocol": "figure7_imagenet1k_complete_val_v1",
        "dataset_manifest_sha256": dataset_sha,
        "dataset_image_count": EXPECTED_IMAGES,
        "classifier": settings["classifier"],
        "classifier_is_paper_specified": False,
        "intact": intact,
        "channel": settings.get("channel"),
        "loss_axis_rate_policy": (
            "No common byte cap is specified for Figure 7(a) in the paper. "
            "This replay does not enforce equal transmitted bytes across codecs; "
            "per-method actual wire bytes are reported at every point."),
        "kdc_val_session_policy": "independent_images_train_only_kb_no_session_update",
        "methods_evaluated": {axis: list(expected_methods[axis])
                              for axis in ("loss", "budget")},
        "methods_unavailable": {
            "loss": [method for method in LOSS_METHODS
                     if method not in expected_methods["loss"]],
            "budget": [method for method in BUDGET_METHODS
                       if method not in expected_methods["budget"]],
        },
        "curves": {},
    }
    for axis in ("loss", "budget"):
        report["curves"][axis] = {}
        for method in expected_methods[axis]:
            entries = []
            for x in sorted(expected_points[axis], key=float):
                key = axis, x, method
                if seen[key] != set(expected_targets):
                    raise ValueError(f"{method}/{axis}/{x} 未覆盖完整且相同的 50,000 张验证图")
                hits = correct[key]
                low, high = wilson_95(hits, EXPECTED_IMAGES)
                sizes = sent_sizes[key]
                entry = {"x": float(x) if axis == "loss" else int(x),
                                "top1": hits / EXPECTED_IMAGES,
                                "hits": hits, "n": EXPECTED_IMAGES,
                                "ci95": [low, high],
                                "mean_sent_bytes": statistics.fmean(sizes),
                                "median_sent_bytes": statistics.median(sizes),
                                "min_sent_bytes": min(sizes),
                                "max_sent_bytes": max(sizes)}
                if method in ("jpeg", "prog_jpeg"):
                    qualities = jpeg_qualities[key]
                    codec_sizes = jpeg_codec_sizes[key]
                    entry.update({
                        "jpeg_quality_range": ([min(qualities), max(qualities)]
                                               if qualities else None),
                        "jpeg_quality_defined_count": len(qualities),
                        "jpeg_quality_missing_count": jpeg_quality_missing[key],
                        "min_codec_bytes": min(codec_sizes),
                        "max_codec_bytes": max(codec_sizes),
                        "mean_codec_bytes": statistics.fmean(codec_sizes),
                    })
                entries.append(entry)
            report["curves"][axis][method] = entries
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    colors = {"jpeg": "#7f8a9b", "prog_jpeg": "#2c88ba", "limitnet": "#e7a300",
              "starfish": "#45a265", "kdc": "#c9353e"}
    labels = {"jpeg": "JPEG", "prog_jpeg": "ProgJPEG", "limitnet": "LimitNet",
              "starfish": "StarFish", "kdc": "KDC"}
    for axis in ("loss", "budget"):
        figure, plot = plt.subplots(figsize=(8, 5.5))
        for method in expected_methods[axis]:
            entries = report["curves"][axis][method]
            xx = [entry["x"] * (100 if axis == "loss" else 1 / 1000)
                  for entry in entries]
            yy = [entry["top1"] for entry in entries]
            lower = [entry["top1"] - entry["ci95"][0] for entry in entries]
            upper = [entry["ci95"][1] - entry["top1"] for entry in entries]
            plot.errorbar(xx, yy, yerr=[lower, upper], marker="o", linewidth=2,
                          capsize=3, color=colors[method], label=labels[method])
        plot.set_xlabel("Packet loss rate (%)" if axis == "loss"
                        else "Transmission budget (kB)")
        plot.set_ylabel("Top-1 accuracy")
        plot.set_ylim(0, 1)
        plot.grid(alpha=0.2)
        plot.legend(frameon=False)
        plot.set_title(f"Figure 7({'a' if axis == 'loss' else 'b'}) · ImageNet-1K val (50,000)")
        figure.tight_layout()
        figure.savefig(output_dir / f"figure7_{axis}.png", dpi=220)
        plt.close(figure)
    return report
