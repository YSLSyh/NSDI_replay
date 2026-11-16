"""合并同一批 300 张 ImageNet 图的 Figure 7 小样本曲线并绘图。

StarFish 无可核验的 ImageNet 权重，不能用占位数值补齐曲线。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from run_figure7_kdc_pilot import _atomic_json


METHODS = {"loss": ("prog_jpeg", "limitnet", "kdc"),
           "budget": ("jpeg", "prog_jpeg", "limitnet", "kdc")}
COLORS = {"jpeg": "#ab8bc7", "prog_jpeg": "#df7078",
          "limitnet": "#58b5a6", "kdc": "#70a9d2"}
LABELS = {"jpeg": "JPEG", "prog_jpeg": "ProgJPEG",
          "limitnet": "LimitNet", "kdc": "KDC"}


def _point(axis: str, value: object) -> float | int:
    if axis == "loss":
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("丢包率点位无效")
        return float(format(number, ".8g"))
    number = int(value)
    if number != value or number <= 0:
        raise ValueError("预算点位无效")
    return number


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _load_new(root: Path) -> list[tuple[dict, list[dict]]]:
    loaded = []
    for settings_path in sorted(root.rglob("settings.json")):
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings.get("protocol") != "figure7_cross_class_300_point_v1":
            continue
        runs = settings_path.with_name("runs.jsonl")
        if not runs.is_file():
            continue
        rows = _read_rows(runs)
        if len(rows) != len(settings["sample_ids"]):
            continue
        loaded.append((settings, rows))
    return loaded


def _load_legacy(path: Path | None, method: str, loss_rate: float) -> tuple[dict, list[dict]] | None:
    if path is None:
        return None
    rows = _read_rows(path)
    if len(rows) != 300:
        raise ValueError(f"旧版 {method} 10 kB 结果不是 300 张")
    for row in rows:
        if row["method"] != method or row.get("channel_event_id") is None:
            raise ValueError(f"旧版 {method} 结果不是配对信道事件")
        if method == "kdc":
            if row.get("axis") != "budget" or row.get("x") != 10_000:
                raise ValueError("旧版 KDC 结果不是 10 kB 预算点")
        else:
            if row.get("budget_bytes") != 10_000 or row.get("loss_rate") != loss_rate:
                raise ValueError("旧版 JPEG 结果不是指定预算或丢包率")
            row["axis"], row["x"] = "budget", 10_000
    settings = {
        "axis": "budget", "point": 10_000, "method": method,
        "loss_rate": loss_rate, "sample_count": 300, "num_shards": 1,
        "shard_index": 0, "sample_ids": [row["image_id"] for row in rows],
        "global_sample_ids": [row["image_id"] for row in rows],
        "channel_seed": 2026, "sample_seed": 2026,
        "dataset_manifest_sha256": "legacy_verified_by_paired_ids",
    }
    return settings, rows


def _aggregate(items: list[tuple[dict, list[dict]]], args) -> dict:
    from replay.evaluation.figure7_classification import wilson_95

    groups: dict[tuple[str, float | int, str], list[tuple[dict, list[dict]]]] = defaultdict(list)
    for settings, rows in items:
        axis = settings["axis"]
        point = _point(axis, settings["point"])
        method = settings["method"]
        if method not in METHODS.get(axis, ()):
            raise ValueError(f"非法方法或坐标：{method}/{axis}")
        groups[axis, point, method].append((settings, rows))
    expected = {
        "loss": tuple(_point("loss", x) for x in args.loss_rates),
        "budget": tuple(_point("budget", x) for x in args.budgets_bytes),
    }
    required = [(axis, point, method) for axis in expected for point in expected[axis]
                for method in METHODS[axis]]
    missing = [f"{axis}/{point}/{method}" for axis, point, method in required
               if (axis, point, method) not in groups]
    if missing:
        raise ValueError("缺少完整点位：" + ", ".join(missing))
    reference_ids: set[str] | None = None
    targets: dict[str, int] = {}
    intact: dict[str, int] = {}
    events: dict[tuple[str, float | int, str], str] = {}
    curves = {"loss": {}, "budget": {}}
    for axis, point, method in required:
        shards = groups[axis, point, method]
        by_id = {}
        for settings, rows in shards:
            if (settings.get("sample_count") != 300 or
                    settings.get("channel_seed") != 2026 or
                    settings.get("sample_seed") != 2026 or
                    (axis == "loss" and not math.isclose(settings["loss_rate"], point,
                                                         abs_tol=1e-9)) or
                    (axis == "budget" and not math.isclose(settings["loss_rate"],
                                                           args.budget_loss_rate,
                                                           abs_tol=1e-9))):
                raise ValueError(f"点位设置不匹配：{axis}/{point}/{method}")
            expected_ids = set(settings["sample_ids"])
            if set(row["image_id"] for row in rows) != expected_ids:
                raise ValueError(f"分片缺少图像或有重复：{axis}/{point}/{method}")
            for row in rows:
                image_id = row["image_id"]
                if (image_id in by_id or row["axis"] != axis or
                        _point(axis, row["x"]) != point or
                        row["method"] != method or
                        not 0 <= row["sent_bytes"] <=
                        (point if axis == "budget" else float("inf"))):
                    raise ValueError(f"逐图记录重复、错位或超预算：{axis}/{point}/{method}")
                by_id[image_id] = row
        ids = set(by_id)
        if len(ids) != 300 or (reference_ids is not None and ids != reference_ids):
            raise ValueError(f"方法没有覆盖同一批 300 张：{axis}/{point}/{method}")
        reference_ids = ids
        for image_id, row in by_id.items():
            target = targets.setdefault(image_id, row["target"])
            original = intact.setdefault(image_id, row["intact_prediction"])
            key = (axis, point, image_id)
            event = events.setdefault(key, row["channel_event_id"])
            if (target != row["target"] or original != row["intact_prediction"] or
                    event != row["channel_event_id"]):
                raise ValueError(f"标签、原图预测或配对信道事件不一致：{image_id}")
        rows = list(by_id.values())
        hits = sum(row["prediction"] == row["target"] for row in rows)
        lo, hi = wilson_95(hits, 300)
        entry = {"x": point, "top1": hits / 300, "hits": hits, "n": 300,
                 "ci95": [lo, hi],
                 "mean_sent_bytes": statistics.fmean(row["sent_bytes"] for row in rows),
                 "decode_failures": sum(row.get("status") == "decode_failed" for row in rows),
                 "infeasible_budget": sum(row.get("status") == "infeasible_budget" for row in rows)}
        curves[axis].setdefault(method, []).append(entry)
    for axis in curves:
        for method in curves[axis]:
            curves[axis][method].sort(key=lambda entry: entry["x"])
    intact_hits = sum(intact[i] == targets[i] for i in reference_ids)
    return {
        "protocol": "figure7_cross_class_300_curves_v1", "pilot_only": True,
        "n_images": 300, "distinct_wnids": 300,
        "loss_rates": list(expected["loss"]),
        "budgets_bytes": list(expected["budget"]),
        "budget_axis_loss_rate": args.budget_loss_rate,
        "loss_axis_fixed_kdc_limitnet_budget_bytes": args.fixed_loss_budget_bytes,
        "loss_axis_common_byte_cap": False,
        "intact": {"top1": intact_hits / 300, "hits": intact_hits,
                   "ci95": list(wilson_95(intact_hits, 300))},
        "methods_unavailable": ["starfish"],
        "starfish_reason": "No verifiable ImageNet pretrained checkpoint available",
        "curves": curves,
    }


def _plot(report: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for axis in ("loss", "budget"):
        fig, ax = plt.subplots(figsize=(9, 6))
        if axis == "loss":
            # 论文图 7(a) 是按丢包等级分组的柱状图，不是折线图。
            methods = ("kdc", "prog_jpeg", "limitnet")
            width = .23
            for index, method in enumerate(methods):
                entries = report["curves"][axis][method]
                centers = [group + (index - 1) * width
                           for group in range(len(entries))]
                ax.bar(centers, [row["top1"] for row in entries], width=width,
                       color=COLORS[method] + "55", edgecolor=COLORS[method],
                       linewidth=1.8, label=LABELS[method])
            rates = report["loss_rates"]
            if rates == [.03, .075, .12]:
                tick_labels = ["Low (<5%)\n3%", "Medium (5–10%)\n7.5%",
                               "High (>10%)\n12%"]
            else:
                tick_labels = [f"{rate * 100:g}%" for rate in rates]
            ax.set_xticks(range(len(rates)), tick_labels)
        else:
            # 论文图 7(b) 是码量与 Top-1 的折线图，并标出完整图像参考水平。
            markers = {"jpeg": "D", "prog_jpeg": "s", "limitnet": "^", "kdc": "o"}
            for method in ("kdc", "prog_jpeg", "limitnet", "jpeg"):
                entries = report["curves"][axis][method]
                ax.plot([row["x"] / 1000 for row in entries],
                        [row["top1"] for row in entries], marker=markers[method],
                        linewidth=2.3, markersize=7, color=COLORS[method],
                        label=LABELS[method])
            ax.axhline(report["intact"]["top1"], color="#b42932",
                       linestyle=(0, (5, 4)), linewidth=1.8, zorder=0)
            ax.text(max(report["budgets_bytes"]) / 1000, report["intact"]["top1"] + .015,
                    f"Intact {report['intact']['top1']:.3f}", ha="right",
                    color="#b42932", fontsize=10)
        ax.set_xlabel("Packet loss category" if axis == "loss" else
                      "Transmission budget (kB)", fontsize=14)
        ax.set_ylabel("Top-1 accuracy", fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
        ax.set_title(f"Figure 7({'a' if axis == 'loss' else 'b'}) · ImageNet-1K classification (pilot)",
                     fontsize=16)
        if axis == "loss":
            mean_kb = {method: statistics.fmean(
                row["mean_sent_bytes"] for row in report["curves"][axis][method]) / 1000
                for method in ("kdc", "prog_jpeg", "limitnet")}
            fig.text(.5, .052,
                     "Actual mean wire: " + "; ".join(
                         f"{LABELS[method]} {mean_kb[method]:.1f} kB"
                         for method in ("kdc", "prog_jpeg", "limitnet")),
                     ha="center", fontsize=10, color="#444")
            fig.text(.5, .025,
                     "KDC/LimitNet cap 30 kB; ProgJPEG full Q75; rates unequal · StarFish unavailable",
                     ha="center", fontsize=10, color="#555")
        else:
            lowest = min(report["budgets_bytes"])
            kdc_low = next(row for row in report["curves"]["budget"]["kdc"]
                           if row["x"] == lowest)
            jpeg_low = next(row for row in report["curves"]["budget"]["jpeg"]
                            if row["x"] == lowest)
            fig.text(.5, .046,
                     f"{report['budget_axis_loss_rate']:.0%} independent random packet loss · StarFish unavailable",
                     ha="center", fontsize=10, color="#555")
            fig.text(.5, .018,
                     f"At {lowest/1000:g} kB, budget infeasible: "
                     f"KDC {kdc_low['infeasible_budget']}/{kdc_low['n']}; "
                     f"JPEG {jpeg_low['infeasible_budget']}/{jpeg_low['n']}",
                     ha="center", fontsize=11, color="#333")
        fig.subplots_adjust(left=.12, right=.98, top=.88, bottom=.19)
        fig.savefig(output_dir / f"figure7_{axis}_pilot300.png", dpi=200,
                    facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="包含本轮各方法、点位和分片目录的根目录")
    parser.add_argument("--loss-rates", type=float, nargs="+", required=True)
    parser.add_argument("--budgets-bytes", type=int, nargs="+", required=True)
    parser.add_argument("--budget-loss-rate", type=float, default=.05)
    parser.add_argument("--fixed-loss-budget-bytes", type=int, default=30_000)
    parser.add_argument("--legacy-kdc-10k", type=Path)
    parser.add_argument("--legacy-jpeg-10k", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (len(args.loss_rates) != len(set(args.loss_rates)) or
            len(args.budgets_bytes) != len(set(args.budgets_bytes)) or
            not 0 <= args.budget_loss_rate <= 1):
        parser.error("点位重复或丢包率无效")
    items = _load_new(args.root)
    for path, method in ((args.legacy_kdc_10k, "kdc"),
                         (args.legacy_jpeg_10k, "jpeg")):
        legacy = _load_legacy(path, method, args.budget_loss_rate)
        if legacy is not None:
            items.append(legacy)
    report = _aggregate(items, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "report.json", report)
    _plot(report, args.output_dir)
    print(json.dumps({"report": str(args.output_dir / "report.json"),
                      "fig7a": str(args.output_dir / "figure7_loss_pilot300.png"),
                      "fig7b": str(args.output_dir / "figure7_budget_pilot300.png")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
