"""Cityscapes 图 8(a)：真值重点区域、三类别独立编码、随机丢包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from replay.core.config import load_config
from replay.evaluation.figure8a_gt_random import (
    TARGETS, run_gt_val, summarize_gt_val,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tx_stage12_qwen4b.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=(*TARGETS, "all"), default="all",
                        help="各类别单独编码；两卡可用不同 task 并行")
    parser.add_argument("--limitnet-checkpoint", default=None)
    parser.add_argument("--limitnet-mode", choices=("official224", "native_adapted"),
                        default="native_adapted")
    parser.add_argument("--budget-bytes", type=int, default=30_000)
    parser.add_argument("--sender-profile-config", default=None,
                        help="Cityscapes train 冻结的 frozen_profiles.json；正式 val 评测必填")
    parser.add_argument("--max-images", type=int, default=100,
                        help="先 2 张冒烟或 100 张；同目录改 500 可续跑")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--roi-weight", type=float, default=4.0)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--save-examples", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true",
                        help="三个任务均完成后汇总 IoU/误差棒，不加载 GPU 模型")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    if args.summarize_only:
        result = summarize_gt_val(config.path(args.output),
                                  max_images=args.max_images,
                                  bootstrap=args.bootstrap, seed=args.seed)
    else:
        if args.limitnet_checkpoint is None:
            parser.error("评测必须指定 --limitnet-checkpoint")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("评测需要可用 CUDA GPU")
        result = run_gt_val(
            config, output=args.output,
            limitnet_checkpoint=args.limitnet_checkpoint,
            task=args.task, limitnet_mode=args.limitnet_mode,
            device=device, budget_bytes=args.budget_bytes,
            max_images=args.max_images, seed=args.seed,
            num_shards=args.num_shards, shard_index=args.shard_index,
            roi_weight=args.roi_weight,
            background_weight=args.background_weight,
            save_examples=args.save_examples,
            sender_profile_config=args.sender_profile_config,
        )
        if args.task == "all" and args.num_shards == 1:
            result["merged"] = summarize_gt_val(
                config.path(args.output), max_images=args.max_images,
                bootstrap=args.bootstrap, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
