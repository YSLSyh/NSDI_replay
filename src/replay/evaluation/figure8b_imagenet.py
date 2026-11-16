"""论文图 8(b) 的 ImageNet 分类分辨率实验；独立于 Cityscapes 改编图。"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from PIL import Image

from replay.evaluation.figure7_classification import ClassificationCodec, ValSample


RESOLUTIONS = (64, 128, 256, 512, 1024)
METHODS = ("limitnet", "kdc")


def resize_center_crop(image: Image.Image, resolution: int) -> Image.Image:
    """短边缩放后中心裁成正方形；避免拉伸原图中的物体。"""
    if resolution <= 0:
        raise ValueError("输入分辨率必须为正")
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = resolution / min(width, height)
    resized = rgb.resize((max(resolution, round(width * scale)),
                          max(resolution, round(height * scale))),
                         Image.Resampling.BICUBIC)
    left = (resized.width - resolution) // 2
    top = (resized.height - resolution) // 2
    return resized.crop((left, top, left + resolution, top + resolution))


def psnr_rgb(reference: Image.Image, reconstructed: Image.Image) -> float:
    if reference.size != reconstructed.size:
        raise ValueError("PSNR 必须比较相同尺寸的图像")
    original = np.asarray(reference.convert("RGB"), dtype=np.float64)
    result = np.asarray(reconstructed.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((original - result) ** 2))
    return math.inf if mse == 0 else float(10 * math.log10(255.0 ** 2 / mse))


def pilot_samples(samples: list[ValSample], *, count: int | None,
                  seed: int) -> list[ValSample]:
    """只有显式指定 count 才抽样；正式实验使用全部 50,000 张。"""
    if count is None:
        return samples
    if not 0 < count < len(samples):
        raise ValueError("pilot-images 必须介于 1 和验证集图像总数减 1 之间")
    indices = sorted(random.Random(seed).sample(range(len(samples)), count))
    return [samples[index] for index in indices]


def channel_event_id(seed: int, resolution: int, image_id: str) -> str:
    return sha256(f"{seed}\t{resolution}\t{image_id}".encode()).hexdigest()


def evaluate_image_resolution(
    *, sample: ValSample, image: Image.Image, resolution: int,
    classifier: Any, codecs: Mapping[str, ClassificationCodec],
    budget_bytes: int, loss_rate: float, channel_seed: int,
) -> list[dict[str, Any]]:
    """同图同分辨率共用逐包随机数；Top-1 对解码失败计错。"""
    if budget_bytes <= 0 or not 0 <= loss_rate <= 1:
        raise ValueError("线字节预算或丢包率无效")
    if set(codecs) != set(METHODS):
        raise ValueError("图 8(b) 必须同时包含 KDC 和 LimitNet")
    source = resize_center_crop(image, resolution)
    intact_prediction = classifier.predict(source)
    event_id = channel_event_id(channel_seed, resolution, sample.image_id)
    event_seed = int(event_id[:16], 16)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        codec = codecs[method]
        if codec.name != method:
            raise ValueError(f"编解码器方法名不一致：{method}")
        base: dict[str, Any] = {
            "image_id": sample.image_id, "resolution": resolution,
            "method": method, "target": sample.target,
            "intact_prediction": intact_prediction,
            "channel_event_id": event_id, "loss_rate": loss_rate,
            "budget_bytes": budget_bytes,
        }
        try:
            payload, metadata = codec.encode(source, budget_bytes=budget_bytes)
        except ValueError as error:
            message = str(error)
            # 预算不可行是实验结果；其他 ValueError 是实现错误，直接暴露。
            if "预算" not in message or not any(
                    word in message for word in ("无法满足", "耗尽")):
                raise
            rows.append({**base, "status": "infeasible_budget",
                         "reason": message, "sent_bytes": 0,
                         "packet_count": 0, "lost_packets": 0,
                         "prediction": -1, "psnr_db": None})
            continue
        sent = metadata.get("wire_bytes")
        packet_count = metadata.get("loss_packet_count")
        if (type(sent) is not int or sent != len(payload) or sent > budget_bytes or
                type(packet_count) is not int or packet_count < 0):
            raise ValueError(f"{method} 实际传输字节或数据包数量不一致")
        if metadata.get("status") == "infeasible_header":
            rows.append({**base, "status": "infeasible_budget",
                         "reason": "LimitNet 控制清单超过预算", "sent_bytes": sent,
                         "packet_count": 0, "lost_packets": 0,
                         "prediction": -1, "psnr_db": None})
            continue
        rng = random.Random(event_seed)
        loss_flags = tuple(rng.random() < loss_rate for _ in range(packet_count))
        result = codec.decode(payload, loss_flags=loss_flags,
                              received_prefix_bytes=None)
        if result is not None and result.size != source.size:
            raise ValueError(f"{method} 解码尺寸与输入分辨率不同")
        score = None if result is None else psnr_rgb(source, result)
        if score is not None and not math.isfinite(score):
            raise ValueError("无损相同图像的 PSNR 为无穷，需单独处理而不能写入 JSON")
        rows.append({**base, "status": "ok" if result is not None else "decode_failed",
                     "sent_bytes": sent, "packet_count": packet_count,
                     "lost_packets": sum(loss_flags),
                     "loss_flags_sha256": sha256(bytes(loss_flags)).hexdigest(),
                     "prediction": -1 if result is None else classifier.predict(result),
                     "psnr_db": score,
                     "codec_input_size": ([224, 224] if method == "limitnet" else
                                          [resolution, resolution]),
                     "codec_metadata": {key: value for key, value in metadata.items()
                                        if key in ("layers_sent", "total_layers",
                                                   "jpeg_quality", "source_codec_bytes",
                                                   "budget_policy")},
                     })
    return rows


def summarize_rows(rows: list[dict[str, Any]], samples: list[ValSample],
                   resolutions: tuple[int, ...]) -> list[dict[str, Any]]:
    expected = {(sample.image_id, resolution, method)
                for sample in samples for resolution in resolutions for method in METHODS}
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["image_id"], row["resolution"], row["method"])
        if key in by_key:
            raise ValueError(f"重复实验记录：{key}")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError(f"图 8(b) 记录不完整：期望 {len(expected)} 条，实际 {len(by_key)} 条")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["resolution"], row["method"])].append(row)
    summary: list[dict[str, Any]] = []
    for resolution in resolutions:
        for method in METHODS:
            current = grouped[(resolution, method)]
            decoded = [row for row in current if row["status"] == "ok"]
            scores = [row["psnr_db"] for row in decoded]
            intact = sum(row["intact_prediction"] == row["target"] for row in current)
            correct = sum(row["prediction"] == row["target"] for row in current)
            summary.append({
                "resolution": resolution, "method": method,
                "n_images": len(current), "n_decoded": len(decoded),
                "n_infeasible_budget": sum(row["status"] == "infeasible_budget"
                                           for row in current),
                "n_decode_failed": sum(row["status"] == "decode_failed"
                                       for row in current),
                "intact_top1": intact / len(current),
                "top1": correct / len(current),
                "psnr_db": float(np.mean(scores)) if scores else None,
                "psnr_denominator": len(scores),
                "decode_success_fraction": len(decoded) / len(current),
                "mean_sent_bytes": float(np.mean([row["sent_bytes"]
                                                   for row in current])),
            })
    return summary


def plot_figure8b(summary: list[dict[str, Any]], output: Path,
                  *, pilot: bool) -> Path:
    """与论文图 8(b) 一样展示 PSNR 和 Top-1 双纵轴、两种方法。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"limitnet": "#e89b29", "kdc": "#287bb4"}
    labels = {"limitnet": "LimitNet", "kdc": "KDC"}
    markers = {"limitnet": "s", "kdc": "o"}
    fig, ax_psnr = plt.subplots(figsize=(9.5, 5.5))
    ax_top1 = ax_psnr.twinx()
    method_handles = []
    for method in METHODS:
        current = sorted((row for row in summary if row["method"] == method),
                         key=lambda row: row["resolution"])
        valid = [row for row in current if row["psnr_db"] is not None]
        psnr_line, = ax_psnr.plot([row["resolution"] for row in valid],
                                  [row["psnr_db"] for row in valid],
                                  color=colors[method], marker=markers[method],
                                  linewidth=2)
        method_handles.append(psnr_line)
        ax_top1.plot([row["resolution"] for row in current],
                     [row["top1"] for row in current], color=colors[method],
                     marker=markers[method], markerfacecolor="white",
                     markeredgewidth=1.8, linestyle="--", linewidth=2)
    resolutions = sorted({row["resolution"] for row in summary})
    ax_psnr.set_xscale("log", base=2)
    ax_psnr.set_xticks(resolutions, [str(value) for value in resolutions])
    ax_psnr.set_xlabel("Raw input resolution (px)")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_top1.set_ylabel("Top-1 accuracy")
    ax_top1.set_ylim(0, 1)
    ax_psnr.grid(alpha=0.25)
    ax_psnr.legend(method_handles, [labels[method] for method in METHODS],
                   frameon=False, loc="best")
    fig.suptitle("Figure 8(b) · ImageNet-1K resolution sweep" +
                 (" · pilot" if pilot else " · full validation"))
    fig.text(0.02, 0.01,
             "Solid filled markers: PSNR (left). Dashed hollow markers: Top-1 (right). "
             "Top-1 counts decoding failures as errors; PSNR uses decoded images only.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    return output
