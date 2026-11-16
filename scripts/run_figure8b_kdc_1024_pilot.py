"""图 8(b) 的单点快速验证：只测 1024 像素 KDC，报告 Top-1 与 PSNR。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import time

from PIL import Image

from run_figure7_kdc_pilot import ROOT, _atomic_json, select_across_classes


RESOLUTION = 1024
PACKET_PAYLOAD_BYTES = 1200
PROTOCOL = "figure8b_kdc_1024_cross_class_pilot_v1"


def _path(value: Path) -> Path:
    return (value if value.is_absolute() else ROOT / value).resolve()


def summarize(rows: list[dict]) -> dict:
    """解码失败计入 Top-1 分母，PSNR 仅对成功解码图像求均值。"""
    from replay.evaluation.figure7_classification import wilson_95

    if not rows:
        raise ValueError("没有可汇总的图像")
    n = len(rows)
    decoded = [row for row in rows if row["status"] == "ok"]
    scores = [row["psnr_db"] for row in decoded]
    intact_hits = sum(row["intact_prediction"] == row["target"] for row in rows)
    kdc_hits = sum(row["prediction"] == row["target"] for row in rows)
    return {
        "n_images": n, "resolution": RESOLUTION,
        "intact_top1": intact_hits / n,
        "intact_hits": intact_hits,
        "intact_ci95": list(wilson_95(intact_hits, n)),
        "kdc_top1": kdc_hits / n,
        "kdc_hits": kdc_hits,
        "kdc_ci95": list(wilson_95(kdc_hits, n)),
        "kdc_psnr_db": statistics.fmean(scores) if scores else None,
        "psnr_denominator": len(scores),
        "decode_failures": sum(row["status"] == "decode_failed" for row in rows),
        "infeasible_budget": sum(row["status"] == "infeasible_budget" for row in rows),
        "mean_sent_bytes": statistics.fmean(row["sent_bytes"] for row in rows),
        "median_sent_bytes": statistics.median(row["sent_bytes"] for row in rows),
        "mean_lost_packets": statistics.fmean(row["lost_packets"] for row in rows),
    }


def plot_report(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    summary = report["summary"]
    fig, (ax_top, ax_psnr) = plt.subplots(1, 2, figsize=(10, 5))
    values = [summary["intact_top1"], summary["kdc_top1"]]
    ax_top.bar(["Original", "KDC restored"], values,
               color=["#78869a", "#c9353e"], width=0.56)
    for index, value in enumerate(values):
        ax_top.text(index, value + 0.025, f"{value:.1%}", ha="center", fontsize=13)
    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("Top-1 accuracy")
    ax_top.set_title("ImageNet classification")
    if summary["kdc_psnr_db"] is not None:
        ax_psnr.bar(["KDC restored"], [summary["kdc_psnr_db"]],
                    color="#c9353e", width=0.45)
        ax_psnr.text(0, summary["kdc_psnr_db"] + 0.5,
                     f"{summary['kdc_psnr_db']:.2f} dB", ha="center", fontsize=13)
        ax_psnr.set_ylim(0, max(35, summary["kdc_psnr_db"] + 4))
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("Against processed 1024 px source")
    for axis in (ax_top, ax_psnr):
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Figure 8(b) KDC-only pilot · 1024 px", fontsize=17)
    fig.text(0.5, 0.02,
             f"{summary['n_images']} distinct-class images · "
             f"{report['budget_bytes'] / 1000:g} kB cap · "
             f"{report['loss_rate']:.0%} random packet loss · "
             "pilot, not the full resolution sweep",
             ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    temporary = path.with_name(path.name + ".part")
    fig.savefig(temporary, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--intact-predictions", type=Path, required=True)
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/tx_stage12_qwen4b.yaml"))
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--bins-cache", type=Path, required=True)
    parser.add_argument("--kb-path", type=Path, required=True)
    parser.add_argument("--receiver-checkpoint", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, default=30_000)
    parser.add_argument("--loss-rate", type=float, default=0.05)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--channel-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.budget_bytes <= 0 or not 0 <= args.loss_rate <= 1:
        parser.error("预算须为正数，丢包率须在 0 到 1 之间")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("分片编号须在 0 到 num-shards - 1 之间")
    output_dir = _path(args.output_dir)
    if not output_dir.is_relative_to(ROOT):
        parser.error("输出目录必须在 replay 项目内")

    import torch
    from replay.evaluation.figure7_classification import (
        Classifier, validate_imagenet_val, validate_intact_predictions,
    )
    from replay.evaluation.figure7_kdc import (
        Figure7KdcSpec, KdcClassificationCodec, missing_kdc_classification_assets,
    )
    from replay.evaluation.figure8b_imagenet import (
        channel_event_id, psnr_rgb, resize_center_crop,
    )

    all_samples, dataset_sha = validate_imagenet_val(_path(args.val_root))
    global_selected = select_across_classes(all_samples, args.sample_count,
                                            args.sample_seed)
    selected = global_selected[args.shard_index::args.num_shards]
    selected_by_id = {sample.image_id: sample for sample in selected}
    intact_path = _path(args.intact_predictions)
    intact_audit = validate_intact_predictions(intact_path, all_samples)
    if intact_audit["backend"] != "torchvision_vit_l_16_swag_e2e":
        raise ValueError("原图预测必须使用本轮 ViT-L 分类器")
    spec = Figure7KdcSpec(_path(args.config), args.task_description,
                          _path(args.bins_cache), _path(args.kb_path),
                          _path(args.receiver_checkpoint), args.budget_bytes)
    issues = missing_kdc_classification_assets(spec)
    if issues:
        raise FileNotFoundError("KDC 资产缺失：" + "；".join(issues))
    settings = {
        "protocol": PROTOCOL, "pilot_only": True,
        "paper_full_resolution_sweep": False,
        "dataset_manifest_sha256": dataset_sha,
        "n_images": len(selected), "total_sample_count": args.sample_count,
        "global_sample_ids": [sample.image_id for sample in global_selected],
        "sample_ids": [sample.image_id for sample in selected],
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "sample_seed": args.sample_seed, "channel_seed": args.channel_seed,
        "resolution": RESOLUTION, "budget_bytes": args.budget_bytes,
        "loss_rate": args.loss_rate,
        "channel": "independent application packet drops, 1200 B payload",
        "input_transform": "RGB; resize short edge bicubic; center crop 1024 px",
        "psnr_reference": "processed raw input at 1024 px",
        "classification_metric": "1000-way Top-1; failures count incorrect",
        "intact_reference": "classifier on processed 1024 px source",
        "task_description": args.task_description,
        "bins_cache": str(spec.bins_cache), "kb_path": str(spec.kb_path),
        "receiver_checkpoint": str(spec.receiver_checkpoint),
        "classifier_weights": str(_path(args.classifier_weights)),
        "classifier_weights_sha256": intact_audit["weights_sha256"],
        "intact_predictions": str(intact_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("已有试跑目录使用不同设置；请指定新目录")
    elif any(output_dir.iterdir()):
        raise FileExistsError("试跑目录已有其他内容；请指定新目录")
    else:
        _atomic_json(settings_path, settings)

    rows_path = output_dir / "runs.jsonl"
    completed: dict[str, dict] = {}
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                image_id = row.get("image_id")
                sample = selected_by_id.get(image_id)
                if (sample is None or image_id in completed or
                        row.get("target") != sample.target or
                        row.get("method") != "kdc" or
                        row.get("resolution") != RESOLUTION or
                        row.get("budget_bytes") != args.budget_bytes or
                        row.get("loss_rate") != args.loss_rate or
                        not 0 <= row.get("sent_bytes", -1) <= args.budget_bytes):
                    raise ValueError("已有逐图结果与本轮试跑不一致")
                completed[image_id] = row

    if len(completed) < len(selected):
        codec = KdcClassificationCodec.from_spec(spec, torch.device(args.device))
        classifier = Classifier(
            backend="torchvision_vit_l_16_swag_e2e",
            weights_path=_path(args.classifier_weights), device=args.device)
        if classifier.weights_sha256 != intact_audit["weights_sha256"]:
            raise ValueError("试跑分类器与原图预测权重不一致")
        started = time.monotonic()
        with rows_path.open("a", encoding="utf-8") as destination:
            for index, sample in enumerate(selected, 1):
                if sample.image_id in completed:
                    continue
                with Image.open(sample.path) as original:
                    source = resize_center_crop(original, RESOLUTION)
                # 原图 Top-1 也用相同的 1024 像素裁剪，避免预处理差异。
                intact_prediction = classifier.predict(source)
                event_id = channel_event_id(args.channel_seed, RESOLUTION,
                                            sample.image_id)
                base = {
                    "image_id": sample.image_id, "target": sample.target,
                    "method": "kdc", "resolution": RESOLUTION,
                    "budget_bytes": args.budget_bytes, "loss_rate": args.loss_rate,
                    "intact_prediction": intact_prediction,
                    "channel_event_id": event_id,
                }
                try:
                    payload, metadata = codec.encode(source,
                                                     budget_bytes=args.budget_bytes)
                except ValueError as error:
                    message = str(error)
                    if "预算" not in message or not any(
                            word in message for word in ("无法满足", "耗尽")):
                        raise
                    row = {**base, "status": "infeasible_budget", "reason": message,
                           "sent_bytes": 0, "packet_count": 0, "lost_packets": 0,
                           "prediction": -1, "psnr_db": None}
                else:
                    sent = metadata.get("wire_bytes")
                    packet_count = metadata.get("loss_packet_count")
                    if (type(sent) is not int or sent != len(payload) or
                            sent > args.budget_bytes or type(packet_count) is not int or
                            packet_count < 0):
                        raise ValueError("KDC 实际字节或数据包数量不一致")
                    rng = random.Random(int(event_id[:16], 16))
                    loss_flags = tuple(rng.random() < args.loss_rate
                                       for _ in range(packet_count))
                    result = codec.decode(payload, loss_flags=loss_flags,
                                          received_prefix_bytes=None)
                    if result is not None and result.size != source.size:
                        raise ValueError("KDC 解码尺寸与输入分辨率不同")
                    score = None if result is None else psnr_rgb(source, result)
                    if score is not None and not math.isfinite(score):
                        raise ValueError("PSNR 非有限值，需单独处理")
                    row = {
                        **base, "status": "ok" if result is not None else "decode_failed",
                        "sent_bytes": sent, "packet_count": packet_count,
                        "lost_packets": sum(loss_flags),
                        "loss_flags_sha256": sha256(bytes(loss_flags)).hexdigest(),
                        "prediction": -1 if result is None else classifier.predict(result),
                        "psnr_db": score,
                        "codec_metadata": {key: value for key, value in metadata.items()
                                           if key in ("jpeg_quality", "source_codec_bytes",
                                                      "budget_policy")},
                    }
                destination.write(json.dumps(row, ensure_ascii=False,
                                             allow_nan=False) + "\n")
                destination.flush()
                completed[sample.image_id] = row
                if index % 10 == 0 or index == len(selected):
                    print(json.dumps({"done": len(completed),
                                      "total": len(selected),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}),
                          flush=True)

    rows = [completed[sample.image_id] for sample in selected]
    report = {
        "protocol": PROTOCOL, "pilot_only": True,
        "n_images": len(rows), "distinct_wnids": len(selected),
        "budget_bytes": args.budget_bytes, "loss_rate": args.loss_rate,
        "summary": summarize(rows), "runs_file": str(rows_path),
    }
    _atomic_json(output_dir / "report.json", report)
    plot_report(report, output_dir / "kdc_1024_top1_psnr.png")
    print(json.dumps({"report": str(output_dir / "report.json"),
                      "figure": str(output_dir / "kdc_1024_top1_psnr.png"),
                      "summary": report["summary"]}, ensure_ascii=False,
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
