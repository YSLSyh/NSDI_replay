"""只运行一个 ImageNet 图 7(b) KDC 预算点；跨类别小样本，不冒充全量曲线。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]


def _path(value: Path) -> Path:
    return (value if value.is_absolute() else ROOT / value).resolve()


def select_across_classes(samples: list, count: int, seed: int) -> list:
    """只用标签抽取互异类别；编码器始终只接收图像和离线任务描述。"""
    groups: dict[str, list] = defaultdict(list)
    for sample in samples:
        groups[sample.wnid].append(sample)
    if not 1 <= count <= len(groups):
        raise ValueError("试跑张数须在 1 到类别数之间，每类只取一张")
    rng = random.Random(seed)
    chosen = [rng.choice(groups[wnid]) for wnid in
              rng.sample(sorted(groups), count)]
    return sorted(chosen, key=lambda sample: sample.image_id)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _pilot_plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = [report["intact"], report["kdc"]]
    labels = ["Original", "KDC restored"]
    colors = ["#78869a", "#c9353e"]
    values = [entry["top1"] for entry in bars]
    errors = [[entry["top1"] - entry["ci95"][0] for entry in bars],
              [entry["ci95"][1] - entry["top1"] for entry in bars]]
    figure, plot = plt.subplots(figsize=(8, 5.6))
    rectangles = plot.bar(labels, values, color=colors, width=0.5)
    plot.errorbar(range(2), values, yerr=errors, fmt="none", color="#202020",
                  linewidth=1.7, capsize=5)
    for rectangle, value in zip(rectangles, values):
        plot.text(rectangle.get_x() + rectangle.get_width() / 2,
                  value + 0.025, f"{value:.1%}", ha="center", fontsize=15)
    plot.set_ylim(0, 1.04)
    plot.set_ylabel("Top-1 accuracy", fontsize=15)
    plot.tick_params(axis="both", labelsize=14)
    plot.grid(axis="y", alpha=0.2)
    plot.set_axisbelow(True)
    plot.set_title("KDC pilot · ImageNet-1K validation sample", fontsize=17)
    figure.text(0.5, 0.055,
                f"{report['n_images']} images from distinct classes · "
                f"{report['budget_bytes'] / 1000:g} kB cap · "
                f"{report['loss_rate']:.0%} random packet loss · "
                f"KDC mean wire {report['kdc']['mean_sent_bytes'] / 1000:.1f} kB",
                ha="center", fontsize=11)
    figure.text(0.5, 0.018, "Pilot only; not the paper's full-validation curve",
                ha="center", fontsize=10, color="#555555")
    figure.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.18)
    temporary = path.with_name(path.name + ".part")
    figure.savefig(temporary, format="png", dpi=200)
    plt.close(figure)
    os.replace(temporary, path)


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
    parser.add_argument("--budget-bytes", type=int, default=10_000)
    parser.add_argument("--loss-rate", type=float, default=0.05)
    parser.add_argument("--sample-count", type=int, default=300,
                        help="所有分片共同抽取的跨类别图像总数")
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
        Classifier, evaluate_codec_point, validate_imagenet_val,
        validate_intact_predictions, wilson_95,
    )
    from replay.evaluation.figure7_kdc import (
        Figure7KdcSpec, KdcClassificationCodec,
        missing_kdc_classification_assets,
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
    intact_prediction = {}
    with intact_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["image_id"] in selected_by_id:
                intact_prediction[row["image_id"]] = row["prediction"]
    if set(intact_prediction) != set(selected_by_id):
        raise ValueError("原图预测缺少试跑样本")

    spec = Figure7KdcSpec(_path(args.config), args.task_description,
                          _path(args.bins_cache), _path(args.kb_path),
                          _path(args.receiver_checkpoint), args.budget_bytes)
    issues = missing_kdc_classification_assets(spec)
    if issues:
        raise FileNotFoundError("KDC 资产缺失：" + "；".join(issues))
    settings = {
        "protocol": "figure7b_kdc_cross_class_pilot_v1",
        "pilot_only": True, "paper_full_val_curve": False,
        "dataset_manifest_sha256": dataset_sha,
        "n_images": len(selected), "distinct_wnids": len(selected),
        "global_sample_ids": [sample.image_id for sample in global_selected],
        "total_sample_count": args.sample_count,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "sample_ids": [sample.image_id for sample in selected],
        "sample_seed": args.sample_seed, "channel_seed": args.channel_seed,
        "budget_bytes": args.budget_bytes, "loss_rate": args.loss_rate,
        "channel": "independent application packet drops, 1200 B payload",
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
                        row.get("axis") != "budget" or
                        row.get("x") != args.budget_bytes or
                        not 0 <= row.get("sent_bytes", -1) <= args.budget_bytes):
                    raise ValueError("已有逐图结果与本轮试跑不一致")
                completed[image_id] = row

    if len(completed) < len(selected):
        device = torch.device(args.device)
        codec = KdcClassificationCodec.from_spec(spec, device)
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
                try:
                    row = evaluate_codec_point(
                        samples=[sample], classifier=classifier,
                        codecs={"kdc": codec}, axis="budget", x=args.budget_bytes,
                        loss_rate=args.loss_rate, packet_payload_bytes=1200,
                        channel_seed=args.channel_seed)[0]
                    row["status"] = ("decoded" if row["prediction"] >= 0
                                     else "decode_failed")
                except ValueError as error:
                    message = str(error)
                    if "预算" not in message or not any(
                            word in message for word in ("无法满足", "耗尽")):
                        raise
                    token = (f"{args.channel_seed}\tbudget\t{args.budget_bytes}"
                             f"\t{sample.image_id}")
                    row = {"axis": "budget", "x": args.budget_bytes,
                           "method": "kdc", "image_id": sample.image_id,
                           "target": sample.target, "prediction": -1,
                           "sent_bytes": 0, "packet_count": 0,
                           "lost_packets": 0, "status": "infeasible_budget",
                           "reason": message,
                           "channel_event_id": sha256(token.encode()).hexdigest()}
                row["intact_prediction"] = intact_prediction[sample.image_id]
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                completed[sample.image_id] = row
                if index % 10 == 0 or index == len(selected):
                    print(json.dumps({"done": len(completed),
                                      "total": len(selected),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}),
                          flush=True)

    rows = [completed[sample.image_id] for sample in selected]
    n = len(rows)
    intact_hits = sum(row["intact_prediction"] == row["target"] for row in rows)
    kdc_hits = sum(row["prediction"] == row["target"] for row in rows)
    mean_bytes = statistics.fmean(row["sent_bytes"] for row in rows)
    report = {
        "protocol": "figure7b_kdc_cross_class_pilot_v1",
        "pilot_only": True, "n_images": n,
        "distinct_wnids": len({sample.wnid for sample in selected}),
        "budget_bytes": args.budget_bytes, "loss_rate": args.loss_rate,
        "intact": {"top1": intact_hits / n, "hits": intact_hits,
                   "ci95": list(wilson_95(intact_hits, n))},
        "kdc": {"top1": kdc_hits / n, "hits": kdc_hits,
                "ci95": list(wilson_95(kdc_hits, n)),
                "mean_sent_bytes": mean_bytes,
                "median_sent_bytes": statistics.median(row["sent_bytes"] for row in rows),
                "decode_failures": sum(row["status"] == "decode_failed" for row in rows),
                "infeasible_budget": sum(row["status"] == "infeasible_budget" for row in rows),
                "mean_lost_packets": statistics.fmean(row["lost_packets"] for row in rows)},
        "runs_file": str(rows_path),
    }
    _atomic_json(output_dir / "report.json", report)
    _pilot_plot(report, output_dir / "kdc_pilot_top1.png")
    print(json.dumps({"report": str(output_dir / "report.json"),
                      "figure": str(output_dir / "kdc_pilot_top1.png"),
                      "summary": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
