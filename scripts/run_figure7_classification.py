"""ImageNet-1K 图 7 评测入口：先核验完整数据和原图分类，再汇总编解码结果。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from replay.evaluation.figure7_classification import (
    Classifier, EXPECTED_IMAGES, evaluate_codec_point, summarize_figure7,
    validate_imagenet_val, validate_intact_predictions,
)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with part.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="验证官方 50,000 张 val 图和 WNID 标签")
    val.add_argument("--val-root", type=Path, required=True)
    intact = sub.add_parser("intact", help="计算原图 Top-1；支持手工划分多 GPU 分片")
    intact.add_argument("--val-root", type=Path, required=True)
    intact.add_argument("--weights", type=Path, required=True)
    intact.add_argument("--output", type=Path, required=True)
    intact.add_argument("--backend", choices=(
        "torchvision_vit_l_16_swag_e2e", "efficientnet_b0"),
        default="torchvision_vit_l_16_swag_e2e")
    intact.add_argument("--device", default="cuda:0")
    intact.add_argument("--start", type=int, default=0)
    intact.add_argument("--stop", type=int, default=EXPECTED_IMAGES)
    intact.add_argument("--batch-size", type=int, default=1,
                        help="原图分类的推理批次；默认 1 保持既有行为")
    merge = sub.add_parser("merge-intact", help="合并原图分类分片并执行完整集精度门槛")
    merge.add_argument("--val-root", type=Path, required=True)
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    jpeg = sub.add_parser("jpeg-point", help="运行固定 Q75 的 JPEG/ProgJPEG 分类点位")
    jpeg.add_argument("--val-root", type=Path, required=True)
    jpeg.add_argument("--weights", type=Path, required=True)
    jpeg.add_argument("--output", type=Path, required=True)
    jpeg.add_argument("--device", default="cuda:0")
    jpeg.add_argument("--axis", choices=("loss", "budget"), required=True)
    jpeg.add_argument("--x", type=float, required=True,
                      help="loss 轴为 0..1 丢包率；budget 轴为整数传输字节")
    jpeg.add_argument("--loss-rate", type=float, required=True,
                      help="budget 轴固定丢包率；loss 轴应与 --x 相同")
    jpeg.add_argument("--channel-seed", type=int, default=2026)
    jpeg.add_argument("--packet-payload-bytes", type=int, default=1200)
    jpeg.add_argument("--packet-header-bytes", type=int, default=0)
    jpeg.add_argument("--start", type=int, default=0)
    jpeg.add_argument("--stop", type=int, default=EXPECTED_IMAGES)
    summary = sub.add_parser("summarize", help="审计配对逐图结果并绘制正式图 7")
    summary.add_argument("--val-root", type=Path, required=True)
    summary.add_argument("--settings", type=Path, required=True)
    summary.add_argument("--trials", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        samples, digest = validate_imagenet_val(args.val_root)
        result = {"images": len(samples), "classes": len({x.wnid for x in samples}),
                  "dataset_manifest_sha256": digest}
    elif args.command == "intact":
        samples, digest = validate_imagenet_val(args.val_root)
        if not 0 <= args.start < args.stop <= len(samples):
            parser.error("--start/--stop 必须落在 0..50000 范围，且 start < stop")
        if args.batch_size <= 0:
            parser.error("--batch-size 必须为正整数")
        classifier = Classifier(
            backend=args.backend, weights_path=args.weights, device=args.device)
        rows = []
        selected = samples[args.start:args.stop]
        for offset in range(0, len(selected), args.batch_size):
            group = selected[offset:offset + args.batch_size]
            images = []
            for sample in group:
                with Image.open(sample.path) as image:
                    images.append(image.convert("RGB"))
            predictions = classifier.predict_batch(images)
            for sample, prediction in zip(group, predictions, strict=True):
                rows.append({"image_id": sample.image_id, "target": sample.target,
                             "prediction": prediction, "backend": classifier.backend,
                             "weights_sha256": classifier.weights_sha256})
        _write_jsonl_atomic(args.output, rows)
        result = {"rows": len(rows), "start": args.start, "stop": args.stop,
                  "output": str(args.output), "backend": classifier.backend,
                  "batch_size": args.batch_size,
                  "weights_sha256": classifier.weights_sha256,
                  "dataset_manifest_sha256": digest,
                  "full_dataset": args.start == 0 and args.stop == len(samples)}
        if result["full_dataset"]:
            result["intact"] = validate_intact_predictions(args.output, samples)
    elif args.command == "merge-intact":
        samples, digest = validate_imagenet_val(args.val_root)
        merged = {}
        for shard in args.shards:
            with shard.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    image_id = row["image_id"]
                    if image_id in merged:
                        raise ValueError(f"原图分类分片出现重复图像：{image_id}")
                    merged[image_id] = row
        if set(merged) != {sample.image_id for sample in samples}:
            raise ValueError("原图分类分片没有覆盖恰好全部 50,000 张验证图")
        _write_jsonl_atomic(args.output, [merged[x.image_id] for x in samples])
        result = validate_intact_predictions(args.output, samples)
        result.update({"output": str(args.output), "dataset_manifest_sha256": digest})
    elif args.command == "jpeg-point":
        from replay.baselines.figure7_jpeg import JpegClassificationCodec

        samples, digest = validate_imagenet_val(args.val_root)
        if not 0 <= args.start < args.stop <= len(samples):
            parser.error("--start/--stop 必须落在 0..50000 范围")
        if args.axis == "loss" and args.x != args.loss_rate:
            parser.error("loss 轴的 --x 必须等于 --loss-rate")
        classifier = Classifier(
            backend="torchvision_vit_l_16_swag_e2e",
            weights_path=args.weights, device=args.device)
        methods = ("prog_jpeg",) if args.axis == "loss" else ("jpeg", "prog_jpeg")
        codecs = {name: JpegClassificationCodec(
            progressive=name == "prog_jpeg", packet_payload_bytes=args.packet_payload_bytes,
            packet_header_bytes=args.packet_header_bytes) for name in methods}
        rows = evaluate_codec_point(
            samples=samples[args.start:args.stop], classifier=classifier,
            codecs=codecs, axis=args.axis, x=args.x, loss_rate=args.loss_rate,
            packet_payload_bytes=args.packet_payload_bytes,
            channel_seed=args.channel_seed)
        _write_jsonl_atomic(args.output, rows)
        result = {"output": str(args.output), "rows": len(rows),
                  "start": args.start, "stop": args.stop,
                  "methods": methods, "axis": args.axis, "x": args.x,
                  "loss_rate": args.loss_rate, "packet_payload_bytes":
                      args.packet_payload_bytes,
                  "classifier_weights_sha256": classifier.weights_sha256,
                  "dataset_manifest_sha256": digest,
                  "formal_complete": args.start == 0 and args.stop == len(samples)}
    else:
        result = summarize_figure7(
            val_root=args.val_root, settings_path=args.settings,
            trials_path=args.trials, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
