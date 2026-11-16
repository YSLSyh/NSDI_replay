"""合并 300 张跨类图像的图 8(b) 五个分辨率；复用已完成的 1024 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from run_figure7_kdc_pilot import ROOT, _atomic_json
from run_figure8b_pilot_sweep import PROTOCOL, PENDING_RESOLUTIONS
from replay.evaluation.figure8b_imagenet import (
    METHODS, RESOLUTIONS, plot_figure8b, summarize_rows,
)


def _path(value: Path) -> Path:
    return (value if value.is_absolute() else ROOT / value).resolve()


def _read_unique(path: Path) -> dict[tuple[str, int, str], dict]:
    result: dict[tuple[str, int, str], dict] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (row["image_id"], row["resolution"], row["method"])
            if key in result:
                raise ValueError(f"重复逐图结果：{key}")
            result[key] = row
    return result


def merge(shard_dirs: list[Path], kdc_1024_runs: Path,
          limitnet_1024_runs: Path, output_dir: Path) -> dict:
    settings_by_index: dict[int, dict] = {}
    pending: dict[tuple[str, int, str], dict] = {}
    for directory in shard_dirs:
        settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
        if settings.get("protocol") != PROTOCOL:
            raise ValueError(f"不是本轮图 8(b) 分片：{directory}")
        index = settings["shard_index"]
        if index in settings_by_index:
            raise ValueError(f"分片 {index} 重复")
        expected_ids = settings["global_sample_ids"][index::settings["num_shards"]]
        if settings["sample_ids"] != expected_ids:
            raise ValueError(f"分片 {index} 与固定抽样顺序不符")
        rows = _read_unique(directory / "runs.jsonl")
        expected_keys = {(image_id, resolution, method)
                         for image_id in expected_ids
                         for resolution in PENDING_RESOLUTIONS
                         for method in METHODS}
        if set(rows) != expected_keys:
            raise ValueError(f"分片 {index} 尚未完成：{len(rows)}/{len(expected_keys)}")
        for key, row in rows.items():
            if (row["budget_bytes"] != settings["budget_bytes"] or
                    row["loss_rate"] != settings["loss_rate"] or
                    not 0 <= row["sent_bytes"] <= settings["budget_bytes"]):
                raise ValueError(f"分片 {index} 的逐图编码条件不一致：{key}")
            if key in pending:
                raise ValueError(f"多个分片出现同一逐图结果：{key}")
            pending[key] = row
        settings_by_index[index] = settings

    if not settings_by_index:
        raise ValueError("没有分片")
    first = settings_by_index[min(settings_by_index)]
    if set(settings_by_index) != set(range(first["num_shards"])):
        raise ValueError("缺少 GPU 分片")
    local_keys = {"sample_ids", "shard_index"}
    shared = {key: value for key, value in first.items() if key not in local_keys}
    for settings in settings_by_index.values():
        if {key: value for key, value in settings.items() if key not in local_keys} != shared:
            raise ValueError("分片数据、模型或信道设置不一致")
    ids = first["global_sample_ids"]
    if len(ids) != first["sample_count"] or len(set(ids)) != len(ids):
        raise ValueError("跨类样本列表重复或不完整")

    reused = {}
    for method, path in (("kdc", kdc_1024_runs),
                         ("limitnet", limitnet_1024_runs)):
        rows = _read_unique(path)
        expected = {(image_id, 1024, method) for image_id in ids}
        if set(rows) != expected:
            raise ValueError(f"既有 1024 {method} 结果与本轮 300 张样本不一致")
        for key, row in rows.items():
            if (row["budget_bytes"] != first["budget_bytes"] or
                    row["loss_rate"] != first["loss_rate"] or
                    not 0 <= row["sent_bytes"] <= first["budget_bytes"]):
                raise ValueError(f"1024 {method} 预算或丢包设置不一致：{key}")
        reused.update(rows)

    all_rows = {**pending, **reused}
    targets: dict[str, int] = {}
    for resolution in RESOLUTIONS:
        for image_id in ids:
            kdc = all_rows[(image_id, resolution, "kdc")]
            limitnet = all_rows[(image_id, resolution, "limitnet")]
            for field in ("target", "intact_prediction", "channel_event_id"):
                if kdc[field] != limitnet[field]:
                    raise ValueError(f"{resolution}px {image_id} 的配对字段 {field} 不同")
            prior = targets.setdefault(image_id, kdc["target"])
            if prior != kdc["target"]:
                raise ValueError("同一图像在不同分辨率的真值标签不一致")

    samples = [SimpleNamespace(image_id=image_id, target=targets[image_id])
               for image_id in ids]
    ordered = [all_rows[(image_id, resolution, method)]
               for image_id in ids for resolution in RESOLUTIONS
               for method in METHODS]
    summary = summarize_rows(ordered, samples, RESOLUTIONS)
    paired_psnr = []
    for resolution in RESOLUTIONS:
        paired = [(all_rows[(image_id, resolution, "kdc")],
                   all_rows[(image_id, resolution, "limitnet")])
                  for image_id in ids
                  if all_rows[(image_id, resolution, "kdc")]["psnr_db"] is not None
                  and all_rows[(image_id, resolution, "limitnet")]["psnr_db"] is not None]
        paired_psnr.append({
            "resolution": resolution, "n": len(paired),
            "kdc_psnr_db": statistics.fmean(k["psnr_db"] for k, _ in paired)
            if paired else None,
            "limitnet_psnr_db": statistics.fmean(l["psnr_db"] for _, l in paired)
            if paired else None,
        })
    report = {
        "protocol": "figure8b_cross_class_300_full_resolution_pilot_v1",
        "pilot_only": True,
        "n_images": len(ids), "n_rows": len(ordered),
        "resolutions": list(RESOLUTIONS),
        "budget_bytes": first["budget_bytes"], "loss_rate": first["loss_rate"],
        "channel": first["channel"],
        "source_shards": [str(directory) for directory in shard_dirs],
        "reused_1024": {"kdc": str(kdc_1024_runs),
                        "limitnet": str(limitnet_1024_runs)},
        "summary": summary, "paired_psnr_by_resolution": paired_psnr,
        "runs_file": str(output_dir / "runs.jsonl"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("合并输出目录已有内容")
    with (output_dir / "runs.jsonl").open("w", encoding="utf-8") as destination:
        for row in ordered:
            destination.write(json.dumps(row, ensure_ascii=False,
                                         allow_nan=False) + "\n")
    _atomic_json(output_dir / "settings.json", shared)
    _atomic_json(output_dir / "report.json", report)
    plot_figure8b(summary, output_dir / "figure8b_kdc_limitnet_300.png", pilot=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--kdc-1024-runs", type=Path, required=True)
    parser.add_argument("--limitnet-1024-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = _path(args.output_dir)
    if not output_dir.is_relative_to(ROOT):
        parser.error("输出目录必须在 replay 项目内")
    report = merge([_path(path) for path in args.input_dirs],
                   _path(args.kdc_1024_runs), _path(args.limitnet_1024_runs),
                   output_dir)
    print(json.dumps({"report": str(output_dir / "report.json"),
                      "figure": str(output_dir / "figure8b_kdc_limitnet_300.png"),
                      "summary": report["summary"]}, ensure_ascii=False,
                     indent=2))


if __name__ == "__main__":
    main()
