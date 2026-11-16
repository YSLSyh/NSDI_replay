"""在指定空闲 GPU 上顺序完成图 7 的 300 张分片点位，失败即停、可续跑。

已有完整的 KDC/JPEG 10 kB 单点时复用；新部署缺少旧结果时直接计算。
StarFish 无可核验权重。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_figure7_pilot_point.py"
MERGER = ROOT / "scripts" / "merge_figure7_pilot_curve.py"


def _path(value: Path) -> Path:
    return (value if value.is_absolute() else ROOT / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/figure7_sweep300"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/figure7_sweep300_merged"))
    parser.add_argument("--val-root", type=Path, default=Path("data/imagenet_val"))
    parser.add_argument("--intact-predictions", type=Path,
                        default=Path("artifacts/figure7_full_20261114/intact_merged.jsonl"))
    parser.add_argument("--classifier-weights", type=Path,
                        default=Path("assets/classification/vit_l_16_swag-4f3808c9.pth"))
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--loss-rates", type=float, nargs="+",
                        default=(.03, .075, .12))
    parser.add_argument("--budgets-bytes", type=int, nargs="+",
                        default=(5_000, 10_000, 20_000, 30_000))
    parser.add_argument("--budget-loss-rate", type=float, default=.05)
    parser.add_argument("--fixed-loss-budget-bytes", type=int, default=30_000)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--shards", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", required=True)
    parser.add_argument("--legacy-kdc-10k", type=Path,
                        default=Path("artifacts/figure7b_kdc_pilot300_10k_merged/runs.jsonl"))
    parser.add_argument("--legacy-jpeg-10k", type=Path,
                        default=Path("artifacts/jpeg420_figure7b_10k_merged/runs.jsonl"))
    parser.add_argument("--merge-when-complete", action="store_true")
    args = parser.parse_args()
    if (args.sample_count != 300 or args.num_shards != 3 or
            len(set(args.shards)) != len(args.shards) or
            any(not 0 <= shard < args.num_shards for shard in args.shards) or
            not 0 <= args.budget_loss_rate <= 1):
        parser.error("本轮曲线必须为 300 张、3 分片；分片或丢包率无效")
    root = _path(args.root)
    output_dir = _path(args.output_dir)
    if not root.is_relative_to(ROOT) or not output_dir.is_relative_to(ROOT):
        parser.error("输出目录必须在 replay 项目内")
    candidates = {"kdc": _path(args.legacy_kdc_10k),
                  "jpeg": _path(args.legacy_jpeg_10k)}
    legacy = {}
    for method, path in candidates.items():
        # 只复用完整的旧结果；缺失或不完整时由本轮分片重新计算 10 kB 点。
        if path.is_file() and sum(1 for _ in path.open(encoding="utf-8")) == 300:
            legacy[method] = path
        else:
            print(f"[Figure 7 pilot] compute {method}/budget/10000; "
                  f"no complete legacy rows at {path}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
    points = [("loss", rate, rate, ("prog_jpeg", "limitnet", "kdc"))
              for rate in args.loss_rates]
    points.extend(("budget", budget, args.budget_loss_rate,
                   ("jpeg", "prog_jpeg", "limitnet", "kdc"))
                  for budget in args.budgets_bytes)
    total = 0
    for axis, point, loss_rate, methods in points:
        for method in methods:
            if axis == "budget" and point == 10_000 and method in legacy:
                print(f"[Figure 7 pilot] reused {method}/budget/10000", flush=True)
                continue
            point_name = (str(point).replace(".", "p") if axis == "loss"
                          else str(point))
            for shard in args.shards:
                out = root / f"{axis}_{point_name}" / f"{method}_s{shard}"
                report = out / "report.json"
                if report.is_file():
                    prior = json.loads(report.read_text(encoding="utf-8"))
                    rows_file = out / "runs.jsonl"
                    row_count = (sum(1 for _ in rows_file.open(encoding="utf-8"))
                                 if rows_file.is_file() else 0)
                    if (prior.get("method") == method and prior.get("axis") == axis and
                            prior.get("point") == point and
                            prior.get("loss_rate") == loss_rate and
                            prior.get("n_images") == 100 and row_count == 100):
                        print(f"[Figure 7 pilot] reused {method}/{axis}/{point}/s{shard}",
                              flush=True)
                        continue
                command = [
                    sys.executable, str(RUNNER), "--method", method,
                    "--axis", axis, "--point", str(point),
                    "--loss-rate", str(loss_rate),
                    "--fixed-loss-budget-bytes", str(args.fixed_loss_budget_bytes),
                    "--val-root", str(_path(args.val_root)),
                    "--intact-predictions", str(_path(args.intact_predictions)),
                    "--classifier-weights", str(_path(args.classifier_weights)),
                    "--task-description", args.task_description,
                    "--sample-count", str(args.sample_count),
                    "--num-shards", str(args.num_shards),
                    "--shard-index", str(shard), "--device", args.device,
                    "--output-dir", str(out),
                ]
                print(f"[Figure 7 pilot] start {method}/{axis}/{point}/s{shard}",
                      flush=True)
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
                total += 1
    print(f"[Figure 7 pilot] finished {total} new shard jobs", flush=True)
    if args.merge_when_complete:
        command = [
            sys.executable, str(MERGER), "--root", str(root),
            "--loss-rates", *(str(x) for x in args.loss_rates),
            "--budgets-bytes", *(str(x) for x in args.budgets_bytes),
            "--budget-loss-rate", str(args.budget_loss_rate),
            "--fixed-loss-budget-bytes", str(args.fixed_loss_budget_bytes),
            "--output-dir", str(output_dir),
        ]
        if "kdc" in legacy:
            command.extend(("--legacy-kdc-10k", str(legacy["kdc"])))
        if "jpeg" in legacy:
            command.extend(("--legacy-jpeg-10k", str(legacy["jpeg"])))
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
