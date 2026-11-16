"""仅用已保存的实验统计重绘四张展示图，不加载数据集或模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merge_figure7_pilot_curve import _plot as plot_figure7
from plot_figure8a_final import plot_distribution as plot_figure8a
from replay.evaluation.figure8b_imagenet import plot_figure8b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("results/reports"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report7 = json.loads((args.reports / "figure7.json").read_text(encoding="utf-8"))
    report8b = json.loads((args.reports / "figure8b.json").read_text(encoding="utf-8"))
    # 图 7(a)/(b) 读取同一份统计报告；这里仅替换最终图名，不改变数据。
    plot_figure7(report7, args.output)
    (args.output / "figure7_loss_pilot300.png").replace(args.output / "figure7a.png")
    (args.output / "figure7_budget_pilot300.png").replace(args.output / "figure7b.png")
    # 图 8(a) 读取逐图 IoU；图 8(b) 读取跨分辨率汇总。
    plot_figure8a(args.reports / "figure8a_per_image_iou.json",
                  args.output / "figure8a.png")
    plot_figure8b(report8b["summary"], args.output / "figure8b.png", pilot=True)
    for name in ("figure7a.png", "figure7b.png", "figure8a.png", "figure8b.png"):
        print(args.output / name)


if __name__ == "__main__":
    main()
