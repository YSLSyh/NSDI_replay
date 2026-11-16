"""在同一批 300 张 ImageNet val 图上运行图 7 的一个方法和一个坐标点。

每类抽取一张图；分片断点续跑。此脚本产生小样本点位，不能称作论文全量曲线。
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import time

from run_figure7_kdc_pilot import ROOT, _atomic_json, select_across_classes


def _path(value: Path) -> Path:
    return (value if value.is_absolute() else ROOT / value).resolve()


def _point(axis: str, value: float) -> float | int:
    if axis == "loss":
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("丢包率点位须位于 0..1")
        return float(format(value, ".8g"))
    if not value.is_integer() or value <= 0:
        raise ValueError("预算点位须为正整数字节")
    return int(value)


def _codec(args, device):
    if args.method in ("jpeg", "prog_jpeg"):
        from replay.baselines.figure7_jpeg import JpegClassificationCodec

        return JpegClassificationCodec(
            progressive=args.method == "prog_jpeg", quality=75,
            subsampling=args.jpeg_subsampling, packet_payload_bytes=1200,
            packet_header_bytes=8)
    if args.method == "limitnet":
        from replay.evaluation.figure7_limitnet import (
            Figure7LimitNetSpec, LimitNetClassificationCodec,
        )

        return LimitNetClassificationCodec.from_spec(Figure7LimitNetSpec(
            _path(args.limitnet_official_raw), _path(args.limitnet_converted),
            args.fixed_loss_budget_bytes, args.limitnet_layers), device)
    from replay.evaluation.figure7_kdc import Figure7KdcSpec, KdcClassificationCodec

    if not args.task_description:
        raise ValueError("KDC 需要离线任务描述 --task-description")
    spec = Figure7KdcSpec(
        _path(args.config), args.task_description, _path(args.bins_cache),
        _path(args.kb_path), _path(args.receiver_checkpoint),
        args.fixed_loss_budget_bytes)
    return KdcClassificationCodec.from_spec(spec, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("kdc", "limitnet", "jpeg", "prog_jpeg"),
                        required=True)
    parser.add_argument("--axis", choices=("loss", "budget"), required=True)
    parser.add_argument("--point", type=float, required=True)
    parser.add_argument("--loss-rate", type=float, required=True)
    parser.add_argument("--fixed-loss-budget-bytes", type=int, default=30_000)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--intact-predictions", type=Path, required=True)
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/tx_stage12_qwen4b.yaml"))
    parser.add_argument("--task-description", default="")
    parser.add_argument("--bins-cache", type=Path,
                        default=Path("artifacts/semantic_bins_imagenet_broad_reaudited_v4.json"))
    parser.add_argument("--kb-path", type=Path, default=Path("artifacts/posterior_kb"))
    parser.add_argument("--receiver-checkpoint", type=Path,
                        default=Path("artifacts/receiver_pilot_fold0_bf16_15ep_post_layernorm.pt"))
    parser.add_argument("--limitnet-official-raw", type=Path,
                        default=Path("assets/external/limitnet_official/LimitNet-ImageNet"))
    parser.add_argument("--limitnet-converted", type=Path,
                        default=Path("assets/external/limitnet_official/limitnet_imagenet_state.pt"))
    parser.add_argument("--limitnet-layers", type=int, default=20)
    parser.add_argument("--jpeg-subsampling", type=int, choices=(0, 2), default=2)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--channel-seed", type=int, default=2026)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    point = _point(args.axis, args.point)
    if (not 0 <= args.loss_rate <= 1 or args.fixed_loss_budget_bytes <= 0 or
            args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards):
        parser.error("丢包率、固定预算或分片编号无效")
    if args.axis == "loss" and point != args.loss_rate:
        parser.error("loss 轴要求 --point 与 --loss-rate 相同")
    if args.axis == "loss" and args.method == "jpeg":
        parser.error("论文图 7(a) 不含普通 JPEG；请使用 ProgJPEG")
    output_dir = _path(args.output_dir)
    if not output_dir.is_relative_to(ROOT):
        parser.error("输出目录必须位于 replay 项目内")

    import torch
    from replay.evaluation.figure7_classification import (
        Classifier, evaluate_codec_point, validate_imagenet_val,
        validate_intact_predictions, wilson_95,
    )

    all_samples, dataset_sha = validate_imagenet_val(_path(args.val_root))
    global_selected = select_across_classes(all_samples, args.sample_count,
                                            args.sample_seed)
    selected = global_selected[args.shard_index::args.num_shards]
    selected_ids = {s.image_id: s for s in selected}
    intact_path = _path(args.intact_predictions)
    intact_audit = validate_intact_predictions(intact_path, all_samples)
    if intact_audit["backend"] != "torchvision_vit_l_16_swag_e2e":
        raise ValueError("原图预测与本轮分类器架构不一致")
    intact = {}
    with intact_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["image_id"] in selected_ids:
                intact[row["image_id"]] = row["prediction"]
    if set(intact) != set(selected_ids):
        raise ValueError("原图预测缺少试跑图像")

    settings = {
        "protocol": "figure7_cross_class_300_point_v1", "pilot_only": True,
        "axis": args.axis, "point": point, "method": args.method,
        "loss_rate": args.loss_rate,
        "fixed_loss_budget_bytes": args.fixed_loss_budget_bytes,
        "budget_policy": ("fixed_codec_full_quality_or_layer_prefix" if args.axis == "loss"
                          else "per_budget_reencode_or_scan_prefix"),
        "sample_count": args.sample_count, "sample_seed": args.sample_seed,
        "channel_seed": args.channel_seed,
        "global_sample_ids": [s.image_id for s in global_selected],
        "sample_ids": [s.image_id for s in selected],
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "dataset_manifest_sha256": dataset_sha,
        "classifier_weights_sha256": intact_audit["weights_sha256"],
        "jpeg_subsampling": args.jpeg_subsampling if args.method in
            ("jpeg", "prog_jpeg") else None,
        "task_description": args.task_description if args.method == "kdc" else None,
        "channel": "independent application packet loss, 1200 B payload; 8 B JPEG/ProgJPEG headers",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("现有试跑目录设置不同")
    elif any(output_dir.iterdir()):
        raise FileExistsError("试跑目录已有其他内容")
    else:
        _atomic_json(settings_path, settings)

    rows_path = output_dir / "runs.jsonl"
    completed = {}
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                image_id = row.get("image_id")
                if (image_id not in selected_ids or image_id in completed or
                        row.get("method") != args.method or row.get("axis") != args.axis or
                        row.get("x") != point or row.get("target") != selected_ids[image_id].target):
                    raise ValueError("逐图断点与设置不一致")
                completed[image_id] = row

    if len(completed) < len(selected):
        device = torch.device(args.device)
        codec = _codec(args, device)
        classifier = Classifier(
            backend="torchvision_vit_l_16_swag_e2e",
            weights_path=_path(args.classifier_weights), device=args.device)
        if classifier.weights_sha256 != intact_audit["weights_sha256"]:
            raise ValueError("分类器权重与完整原图预测不一致")
        started = time.monotonic()
        with rows_path.open("a", encoding="utf-8") as destination:
            for index, sample in enumerate(selected, 1):
                if sample.image_id in completed:
                    continue
                try:
                    row = evaluate_codec_point(
                        samples=[sample], classifier=classifier,
                        codecs={args.method: codec}, axis=args.axis, x=point,
                        loss_rate=args.loss_rate, packet_payload_bytes=1200,
                        channel_seed=args.channel_seed)[0]
                    row["status"] = "decoded" if row["prediction"] >= 0 else "decode_failed"
                    if row["sent_bytes"] == 0:
                        row["status"] = "infeasible_budget"
                except ValueError as error:
                    message = str(error)
                    if not any(token in message for token in
                               ("预算", "耗尽", "质量 1", "quality=1")):
                        raise
                    token = f"{args.channel_seed}\t{args.axis}\t{format(point, '.8g') if args.axis == 'loss' else point}\t{sample.image_id}"
                    row = {
                        "axis": args.axis, "x": point, "method": args.method,
                        "image_id": sample.image_id, "target": sample.target,
                        "prediction": -1, "sent_bytes": 0,
                        "source_codec_bytes": 0, "packet_count": 0,
                        "lost_packets": 0, "status": "infeasible_budget",
                        "reason": message,
                        "channel_event_id": sha256(token.encode()).hexdigest(),
                    }
                row["intact_prediction"] = intact[sample.image_id]
                row["configured_loss_rate"] = args.loss_rate
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                completed[sample.image_id] = row
                if index % 10 == 0 or index == len(selected):
                    print(json.dumps({"done": len(completed), "total": len(selected),
                                      "elapsed_seconds": round(time.monotonic()-started, 1)}),
                          flush=True)

    rows = [completed[s.image_id] for s in selected]
    n = len(rows)
    hits = sum(row["prediction"] == row["target"] for row in rows)
    intact_hits = sum(row["intact_prediction"] == row["target"] for row in rows)
    report = {
        "protocol": settings["protocol"], "pilot_only": True,
        "axis": args.axis, "point": point, "method": args.method,
        "loss_rate": args.loss_rate, "n_images": n,
        "intact": {"top1": intact_hits/n, "hits": intact_hits,
                   "ci95": list(wilson_95(intact_hits, n))},
        "score": {"top1": hits/n, "hits": hits,
                  "ci95": list(wilson_95(hits, n)),
                  "mean_sent_bytes": statistics.fmean(row["sent_bytes"] for row in rows),
                  "decode_failures": sum(row["status"] == "decode_failed" for row in rows),
                  "infeasible_budget": sum(row["status"] == "infeasible_budget" for row in rows)},
        "runs_file": str(rows_path),
    }
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
