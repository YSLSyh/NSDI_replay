"""从已核验的 Cityscapes 逐图结果绘制论文样式的图 8(a)。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TASKS = (("car", "Car"), ("pedestrian", "Pedestrians"),
         ("vegetation", "Vegetation"))
METHODS = (("kdc_restored", "KDC", "#b9d8ec", "#68a5cf"),
           ("limitnet", "LimitNet", "#f8d4b5", "#ec9a61"))


def plot_pooled_ci(source: Path, output: Path) -> Path:
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(row["class"], row["method"]): row for row in rows}
    expected = {(task, method) for task, _ in TASKS
                for method, *_ in METHODS}
    if set(by_key) != expected or len(rows) != len(expected):
        raise ValueError("图 8(a) 必须恰有三个类别、两种方法的汇总结果")

    fig, ax = plt.subplots(figsize=(9.5, 5.9))
    width = .31
    centers = range(len(TASKS))
    for index, (method, label, fill, edge) in enumerate(METHODS):
        values = []
        lower = []
        upper = []
        for task, _ in TASKS:
            row = by_key[(task, method)]
            value = float(row["iou"])
            ci_low = float(row["ci95_low"])
            ci_high = float(row["ci95_high"])
            if not 0 <= ci_low <= value <= ci_high <= 1:
                raise ValueError(f"{task}/{method} 的 IoU 或置信区间无效")
            values.append(value)
            lower.append(value - ci_low)
            upper.append(ci_high - value)
        positions = [center + (index - .5) * width for center in centers]
        bars = ax.bar(positions, values, width, color=fill, edgecolor=edge,
                      linewidth=1.8, label=label,
                      yerr=[lower, upper], capsize=4,
                      error_kw={"ecolor": "#202020", "elinewidth": 1.2,
                                 "capthick": 1.2})
        for bar, value, error in zip(bars, values, upper):
            ax.text(bar.get_x() + bar.get_width() / 2, value + error + .015,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(list(centers), [label for _, label in TASKS])
    ax.set_xlabel("Detection task", fontsize=13)
    ax.set_ylabel("Class IoU", fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 8(a) · Cityscapes task IoU", fontsize=16)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11)
    # 这张备用图展示聚合 IoU；误差棒是对图像做 bootstrap 得到的区间。
    fig.text(.5, .035,
             "Pilot · 30 kB wire budget · synthetic GE packet loss · no RST · GT oracle focus",
             ha="center", fontsize=9.5, color="#555")
    fig.text(.5, .013, "Bars: pooled class IoU; whiskers: image-bootstrap 95% CI.",
             ha="center", fontsize=9.5, color="#555")
    fig.subplots_adjust(left=.1, right=.98, top=.9, bottom=.17)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


def plot_distribution(source: Path, output: Path) -> Path:
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("protocol") != "figure8a_cityscapes_100_synthetic_ge_per_image_iou_v1":
        raise ValueError("逐图 IoU 数据的实验协议不匹配")
    classes = document.get("classes", {})
    if set(classes) != {task for task, _ in TASKS}:
        raise ValueError("图 8(a) 必须包含 car、pedestrian、vegetation")

    fig, ax = plt.subplots(figsize=(9.5, 5.9))
    width = .27
    for group, (task, _) in enumerate(TASKS):
        scores = classes[task].get("scores", {})
        if set(scores) != {"KDC", "LimitNet"}:
            raise ValueError(f"{task} 缺少 KDC 或 LimitNet 的逐图值")
        if not scores["KDC"] or len(scores["KDC"]) != len(scores["LimitNet"]):
            raise ValueError(f"{task} 的两方法样本数不同")
        for index, (label, fill, edge) in enumerate((
                ("KDC", "#b9d8ec", "#68a5cf"),
                ("LimitNet", "#f8d4b5", "#ec9a61"))):
            values = scores[label]
            if any(not isinstance(value, (int, float)) or
                   not math.isfinite(value) or not 0 <= value <= 1
                   for value in values):
                raise ValueError(f"{task}/{label} 含无效 IoU")
            position = group + (index - .5) * .38
            # 原论文图 8(a) 是逐图分布的箱线图；这里直接使用逐图 IoU，
            # 不从聚合均值或置信区间推造四分位数。
            ax.boxplot(values, positions=[position], widths=width,
                       patch_artist=True, manage_ticks=False, whis=1.5,
                       boxprops={"facecolor": fill, "edgecolor": edge,
                                 "linewidth": 1.7},
                       whiskerprops={"color": "#272727", "linewidth": 1.2},
                       capprops={"color": "#272727", "linewidth": 1.2},
                       medianprops={"color": "#151515", "linewidth": 1.8},
                       flierprops={"marker": "o", "markersize": 3,
                                   "markerfacecolor": "white",
                                   "markeredgecolor": edge, "alpha": .8})

    ax.set_xticks(range(len(TASKS)), [label for _, label in TASKS])
    ax.set_xlabel("Detection task", fontsize=13)
    ax.set_ylabel("Per-image class IoU", fontsize=13)
    ax.set_ylim(-.02, 1.02)
    ax.set_title("Figure 8(a) · Cityscapes task IoU", fontsize=16)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="#b9d8ec", edgecolor="#68a5cf",
                             linewidth=1.7, label="KDC"),
                       Patch(facecolor="#f8d4b5", edgecolor="#ec9a61",
                             linewidth=1.7, label="LimitNet")],
              frameon=False, fontsize=11)
    fig.text(.5, .035,
             "Pilot · 30 kB wire budget · synthetic GE packet loss · no RST · GT oracle focus",
             ha="center", fontsize=9.5, color="#555")
    fig.text(.5, .013,
             "Boxplots: per-image IoU on GT-present images; undecodable images count as 0.",
             ha="center", fontsize=9.5, color="#555")
    fig.subplots_adjust(left=.1, right=.98, top=.9, bottom=.17)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-csv", type=Path,
                        default=Path("artifacts/figure8a_gt_ge30k_oracle_420_100/"
                                     "class_iou_n100.csv"))
    parser.add_argument("--distribution", type=Path,
                        default=Path("artifacts/figure8a_gt_ge30k_oracle_420_100/"
                                     "figure8a_paper_box_distribution.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/figure8a_final.png"))
    parser.add_argument("--pooled-output", type=Path,
                        default=Path("artifacts/figure8a_final_pooled_ci.png"))
    args = parser.parse_args()
    print(plot_pooled_ci(args.pooled_csv, args.pooled_output))
    print(plot_distribution(args.distribution, args.output))


if __name__ == "__main__":
    main()
