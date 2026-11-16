"""Cityscapes train 上微调 LimitNet；固定 224 与原生裁块模式分别留痕。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from replay.baselines.limitnet_train import (INPUT_MODES, TRAINING_RECIPES, FineTuneOptions,
                                             finetune_limitnet_cityscapes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes-root", type=Path, required=True)
    parser.add_argument("--segmenter", type=Path, required=True,
                        help="本地 Cityscapes 19 类 SegFormer-B5 权重目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="可选：由 convert_limitnet_official.py 转换的官方权重")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="native_crop",
                        help="official224_full_frame 按官方 demo 将整帧缩放到 224×224；"
                             "native_crop 保持原生像素裁块")
    parser.add_argument("--training-recipe", choices=TRAINING_RECIPES,
                        default="cityscapes_multiloss",
                        help="official_phase3_segmentation 按官方第三阶段冻结显著性分支、"
                             "只用任务损失；Cityscapes 分割头仍是项目适配")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="默认：项目多损失 1e-4，官方第三阶段式 5e-5")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=5.0)
    parser.add_argument("--saliency-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    options = FineTuneOptions(
        cityscapes_root=args.cityscapes_root, segmenter_path=args.segmenter,
        output_checkpoint=args.output, init_checkpoint=args.init_checkpoint,
        epochs=args.epochs, max_images=args.max_images, crop_size=args.crop_size,
        learning_rate=args.learning_rate, accumulation=args.accumulation,
        task_weight=args.task_weight, reconstruction_weight=args.reconstruction_weight,
        saliency_weight=args.saliency_weight, seed=args.seed,
        input_mode=args.input_mode, training_recipe=args.training_recipe,
    )
    report = finetune_limitnet_cityscapes(options, torch.device(args.device))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
