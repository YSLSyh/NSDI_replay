"""图 8(b) 300 张跨类试跑：只计算尚未完成的 64–512 像素点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from PIL import Image

from run_figure7_kdc_pilot import ROOT, _atomic_json, select_across_classes


PENDING_RESOLUTIONS = (64, 128, 256, 512)
PROTOCOL = "figure8b_cross_class_300_four_pending_resolutions_v1"


def _path(value: Path) -> Path:
    return (value if value.is_absolute() else ROOT / value).resolve()


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
    parser.add_argument("--limitnet-official-raw", type=Path, required=True)
    parser.add_argument("--limitnet-converted", type=Path, required=True)
    parser.add_argument("--limitnet-layers", type=int, default=20)
    parser.add_argument("--budget-bytes", type=int, default=30_000)
    parser.add_argument("--loss-rate", type=float, default=0.05)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--channel-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.budget_bytes <= 0 or not 0 <= args.loss_rate <= 1 or
            args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards):
        parser.error("预算、丢包率或分片编号无效")
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
    from replay.evaluation.figure7_limitnet import (
        Figure7LimitNetSpec, LimitNetClassificationCodec,
        verify_official_imagenet_limitnet,
    )
    from replay.evaluation.figure8b_imagenet import evaluate_image_resolution

    all_samples, dataset_sha = validate_imagenet_val(_path(args.val_root))
    global_selected = select_across_classes(all_samples, args.sample_count,
                                            args.sample_seed)
    selected = global_selected[args.shard_index::args.num_shards]
    selected_by_id = {sample.image_id: sample for sample in selected}
    intact_path = _path(args.intact_predictions)
    intact_audit = validate_intact_predictions(intact_path, all_samples)
    if intact_audit["backend"] != "torchvision_vit_l_16_swag_e2e":
        raise ValueError("原图分类器与当前评测不一致")
    spec_kdc = Figure7KdcSpec(
        _path(args.config), args.task_description, _path(args.bins_cache),
        _path(args.kb_path), _path(args.receiver_checkpoint), args.budget_bytes)
    issues = missing_kdc_classification_assets(spec_kdc)
    if issues:
        raise FileNotFoundError("KDC 资产缺失：" + "；".join(issues))
    spec_limitnet = Figure7LimitNetSpec(
        _path(args.limitnet_official_raw), _path(args.limitnet_converted),
        args.budget_bytes, args.limitnet_layers)
    provenance = verify_official_imagenet_limitnet(spec_limitnet)

    settings = {
        "protocol": PROTOCOL, "pilot_only": True,
        "dataset_manifest_sha256": dataset_sha,
        "global_sample_ids": [sample.image_id for sample in global_selected],
        "sample_ids": [sample.image_id for sample in selected],
        "sample_count": args.sample_count,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "sample_seed": args.sample_seed, "channel_seed": args.channel_seed,
        "resolutions": list(PENDING_RESOLUTIONS),
        "reuse_resolution": 1024,
        "budget_bytes": args.budget_bytes, "loss_rate": args.loss_rate,
        "channel": "independent application packet drops, 1200 B payload",
        "input_transform": "RGB; resize short edge bicubic; center crop square",
        "psnr_reference": "processed raw input at the same resolution",
        "classifier_weights": str(_path(args.classifier_weights)),
        "classifier_weights_sha256": intact_audit["weights_sha256"],
        "intact_predictions": str(intact_path),
        "kdc_config": str(spec_kdc.config_path),
        "task_description": spec_kdc.task_description,
        "bins_cache": str(spec_kdc.bins_cache),
        "kb_path": str(spec_kdc.kb_path),
        "receiver_checkpoint": str(spec_kdc.receiver_checkpoint),
        "limitnet_official_raw": str(spec_limitnet.official_raw_checkpoint),
        "limitnet_converted": str(spec_limitnet.converted_checkpoint),
        "limitnet_layers": spec_limitnet.layers,
        "limitnet_provenance": provenance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("已有分片使用不同设置")
    elif any(output_dir.iterdir()):
        raise FileExistsError("输出目录已有其他内容")
    else:
        _atomic_json(settings_path, settings)

    rows_path = output_dir / "runs.jsonl"
    completed: dict[tuple[str, int, str], dict] = {}
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                key = (row.get("image_id"), row.get("resolution"), row.get("method"))
                sample = selected_by_id.get(key[0])
                if (sample is None or key in completed or
                        key[1] not in PENDING_RESOLUTIONS or
                        key[2] not in ("kdc", "limitnet") or
                        row.get("target") != sample.target or
                        row.get("budget_bytes") != args.budget_bytes or
                        row.get("loss_rate") != args.loss_rate or
                        not 0 <= row.get("sent_bytes", -1) <= args.budget_bytes):
                    raise ValueError("已有逐图结果与本轮设置不一致")
                completed[key] = row

    total = len(selected) * len(PENDING_RESOLUTIONS) * 2
    if len(completed) < total:
        device = torch.device(args.device)
        classifier = Classifier(
            backend="torchvision_vit_l_16_swag_e2e",
            weights_path=_path(args.classifier_weights), device=args.device)
        if classifier.weights_sha256 != intact_audit["weights_sha256"]:
            raise ValueError("分类器权重与原图预测审计不一致")
        codecs = {
            "kdc": KdcClassificationCodec.from_spec(spec_kdc, device),
            "limitnet": LimitNetClassificationCodec.from_spec(spec_limitnet, device),
        }
        started = time.monotonic()
        with rows_path.open("a", encoding="utf-8") as destination:
            for index, sample in enumerate(selected, 1):
                if all((sample.image_id, resolution, method) in completed
                       for resolution in PENDING_RESOLUTIONS
                       for method in ("kdc", "limitnet")):
                    continue
                with Image.open(sample.path) as original:
                    image = original.convert("RGB")
                for resolution in PENDING_RESOLUTIONS:
                    if all((sample.image_id, resolution, method) in completed
                           for method in ("kdc", "limitnet")):
                        continue
                    rows = evaluate_image_resolution(
                        sample=sample, image=image, resolution=resolution,
                        classifier=classifier, codecs=codecs,
                        budget_bytes=args.budget_bytes, loss_rate=args.loss_rate,
                        channel_seed=args.channel_seed)
                    for row in rows:
                        key = (row["image_id"], row["resolution"], row["method"])
                        if key in completed:
                            continue
                        destination.write(json.dumps(row, ensure_ascii=False,
                                                     allow_nan=False) + "\n")
                        destination.flush()
                        completed[key] = row
                if index % 10 == 0 or index == len(selected):
                    print(json.dumps({"images_seen": index,
                                      "images_total": len(selected),
                                      "rows_complete": len(completed),
                                      "rows_total": total,
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}),
                          flush=True)

    if len(completed) != total:
        raise ValueError("分片结果不完整")
    report = {"protocol": PROTOCOL, "pilot_only": True,
              "n_images": len(selected), "resolutions": list(PENDING_RESOLUTIONS),
              "n_rows": len(completed), "runs_file": str(rows_path)}
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
