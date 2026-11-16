"""只用 Cityscapes train 为图 8(a) 三任务冻结 30 kB 预量化参数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay.core.config import load_config
from replay.evaluation.tune_gt_30kb_sender import tune_gt_30kb_sender


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tx_stage12_qwen4b.yaml")
    parser.add_argument("--output", default="artifacts/figure8a_gt_train_30kb_tuning")
    parser.add_argument("--positive-per-task", type=int, default=4)
    parser.add_argument("--negative-per-task", type=int, default=1)
    parser.add_argument("--low-frequency-strengths", nargs="+", type=float,
                        default=[16.0, 64.0, 256.0])
    parser.add_argument("--subsamplings", nargs="+", type=int, choices=(0, 2),
                        default=[0],
                        help="0=4:4:4；2=4:2:0。默认不改现有色度采样")
    parser.add_argument("--high-frequency-strength", type=float, default=8.0)
    parser.add_argument("--zero-threshold-multiplier", type=float, default=2.0)
    parser.add_argument("--budget-bytes", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-quality", type=int, default=95)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    report = tune_gt_30kb_sender(
        config, output=args.output,
        positive_per_task=args.positive_per_task,
        negative_per_task=args.negative_per_task,
        low_frequency_strengths=tuple(args.low_frequency_strengths),
        subsamplings=tuple(args.subsamplings),
        high_frequency_strength=args.high_frequency_strength,
        zero_threshold_multiplier=args.zero_threshold_multiplier,
        budget_bytes=args.budget_bytes, seed=args.seed,
        max_quality=args.max_quality)
    print(json.dumps({"path": str(config.path(args.output) / "frozen_profiles.json"),
                      "selected_by_task": report["selected_by_task"],
                      "all_train_samples_feasible":
                      report["all_train_samples_feasible"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
