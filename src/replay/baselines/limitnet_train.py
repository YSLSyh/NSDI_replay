"""以 Cityscapes train 微调 LimitNet；原生裁块与官方固定 224 输入分开记录。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from replay.baselines.limitnet import LimitNetCore, MEAN, STD, priority_map, priority_mask
from replay.data.cityscapes import cityscapes_pairs, load_cityscapes_pair


TARGET_IDS = (13, 11, 8)  # 图 8(a)：car、pedestrian、vegetation 的 Cityscapes trainId
INPUT_MODES = ("native_crop", "official224_full_frame")
TRAINING_RECIPES = ("cityscapes_multiloss", "official_phase3_segmentation")


@dataclass(frozen=True)
class FineTuneOptions:
    cityscapes_root: Path
    segmenter_path: Path
    output_checkpoint: Path
    init_checkpoint: Path | None = None
    epochs: int = 2
    max_images: int | None = None
    crop_size: int = 512
    learning_rate: float | None = None
    accumulation: int = 4
    task_weight: float = 1.0
    reconstruction_weight: float = 5.0
    saliency_weight: float = 0.1
    seed: int = 2026
    input_mode: str = "native_crop"
    training_recipe: str = "cityscapes_multiloss"


def _native_crop(image: Image.Image, label: np.ndarray, crop_size: int,
                 rng: random.Random) -> tuple[Image.Image, np.ndarray]:
    """按原图坐标裁剪；优先覆盖目标类别，绝不缩放训练像素。"""
    width, height = image.size
    if crop_size > min(width, height):
        raise ValueError("裁块不能大于原图最短边")
    if rng.random() < 0.75:
        available = [class_id for class_id in TARGET_IDS if np.any(label == class_id)]
    else:
        available = []
    if available:
        class_id = rng.choice(available)
        yy, xx = np.nonzero(label == class_id)
        sample = rng.randrange(len(xx))
        x = max(0, min(width - crop_size, int(xx[sample]) - crop_size // 2))
        y = max(0, min(height - crop_size, int(yy[sample]) - crop_size // 2))
    else:
        x = rng.randint(0, width - crop_size)
        y = rng.randint(0, height - crop_size)
    return image.crop((x, y, x + crop_size, y + crop_size)), label[y:y + crop_size,
                                                                    x:x + crop_size].copy()


def _training_sample(image: Image.Image, label: np.ndarray, options: FineTuneOptions,
                     rng: random.Random) -> tuple[Image.Image, np.ndarray]:
    """固定尺寸基线照官方 demo 缩放整帧；仅 LimitNet 使用此路径。"""
    if options.input_mode == "native_crop":
        return _native_crop(image, label, options.crop_size, rng)
    if options.input_mode != "official224_full_frame":
        raise ValueError(f"未知的 LimitNet 训练输入模式：{options.input_mode}")
    if label.shape != (image.height, image.width):
        raise ValueError("Cityscapes 图像与标签尺寸不一致")
    sample = image.resize((224, 224), Image.Resampling.BILINEAR)
    target = Image.fromarray(label.astype(np.uint8)).resize(
        (224, 224), Image.Resampling.NEAREST)
    return sample, np.asarray(target, dtype=np.uint8).copy()


def _official_phase3_keep_mask(saliency: torch.Tensor,
                               keep_fraction: float) -> torch.Tensor:
    """官方第三阶段用优先级分位数清零潜变量；兼容动态潜变量尺寸。"""
    if not 0.0 < keep_fraction < 1.0:
        raise ValueError("官方第三阶段训练保留比例必须位于 (0,1)")
    priority = priority_map(saliency).float()
    threshold = torch.quantile(priority.flatten(), 1.0 - keep_fraction)
    return (priority >= threshold).to(dtype=saliency.dtype)


def _to_tensors(image: Image.Image, label: np.ndarray, device: torch.device
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    pixels = pixels.unsqueeze(0).to(device, non_blocking=True).float().div_(255)
    truth = torch.from_numpy(label.astype(np.int64)).unsqueeze(0).to(device, non_blocking=True)
    mean = pixels.new_tensor(MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(STD).view(1, 3, 1, 1)
    return pixels, (pixels - mean) / std, truth


def _initial_core(path: Path | None) -> tuple[LimitNetCore, str]:
    model = LimitNetCore()
    if path is None:
        return model, "cityscapes_from_scratch"
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("预训练权重必须是安全的 state_dict 字典")
    state = value.get("state_dict", value)
    model.load_state_dict({key: tensor for key, tensor in state.items()
                           if key.startswith(("encoder.", "sal_decoder.", "decoder."))},
                          strict=True)
    metadata = value.get("metadata", {})
    previous_source = str(metadata.get("source", "unknown_checkpoint")) if isinstance(metadata, dict) else "unknown_checkpoint"
    if previous_source.startswith("official_limitnet_"):
        source = "cityscapes_finetuned_from_official"
    else:
        source = "cityscapes_finetuned_from_checkpoint"
    return model, source


def finetune_limitnet_cityscapes(options: FineTuneOptions, device: torch.device) -> dict:
    """训练官方网络结构；重点覆盖图 8(a) 三类，同时对完整 19 类做任务损失。"""
    if options.input_mode not in INPUT_MODES:
        raise ValueError(f"未知的 LimitNet 训练输入模式：{options.input_mode}")
    if options.input_mode == "native_crop" and (options.crop_size % 8 or options.crop_size <= 0):
        raise ValueError("crop_size 必须为 8 的正整数倍")
    if options.training_recipe not in TRAINING_RECIPES:
        raise ValueError(f"未知的 LimitNet 训练配方：{options.training_recipe}")
    if (options.training_recipe == "official_phase3_segmentation" and
            options.input_mode != "official224_full_frame"):
        raise ValueError("官方第三阶段式配方要求固定 224×224 整帧输入")
    effective_lr = (options.learning_rate if options.learning_rate is not None else
                    (5e-5 if options.training_recipe == "official_phase3_segmentation"
                     else 1e-4))
    if options.epochs < 1 or options.accumulation < 1 or effective_lr <= 0:
        raise ValueError("epochs、accumulation、learning_rate 必须为正值")
    torch.manual_seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
    pairs = list(cityscapes_pairs(options.cityscapes_root, "train"))
    if not pairs:
        raise ValueError("Cityscapes train 为空")
    rng = random.Random(options.seed)
    if options.max_images is not None:
        if options.max_images < 1:
            raise ValueError("max_images 必须为正值")
        # 固定样本池并记录数量；验证集完全不参与微调。
        rng.shuffle(pairs)
        pairs = pairs[:options.max_images]

    core, source = _initial_core(options.init_checkpoint)
    official_phase3 = options.training_recipe == "official_phase3_segmentation"
    if official_phase3 and source != "cityscapes_finetuned_from_official":
        raise ValueError("官方第三阶段式 Cityscapes 微调必须从官方预训练权重初始化")
    core = core.to(device).train()
    if official_phase3:
        # 官方 train.py 第三阶段冻结显著性分支，任务损失更新编码器和解码器。
        for parameter in core.sal_decoder.parameters():
            parameter.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(options.segmenter_path,
                                                   local_files_only=True)
    segmenter = SegformerForSemanticSegmentation.from_pretrained(
        options.segmenter_path, local_files_only=True).to(device).eval()
    if int(segmenter.config.num_labels) != 19:
        raise ValueError("Cityscapes 微调要求 19 类 SegFormer 分割头")
    for parameter in segmenter.parameters():
        parameter.requires_grad_(False)
    seg_mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    seg_std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    trainable = (parameter for parameter in core.parameters() if parameter.requires_grad)
    optimizer = (torch.optim.Adam(trainable, lr=effective_lr) if official_phase3 else
                 torch.optim.AdamW(trainable, lr=effective_lr))
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else nullcontext
    fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
    history: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(options.epochs):
        rng.shuffle(pairs)
        running = {"task": 0.0, "reconstruction": 0.0, "saliency": 0.0,
                   "total": 0.0}
        for index, (image_path, label_path) in enumerate(pairs):
            image, label = load_cityscapes_pair(image_path, label_path)
            crop, label_crop = _training_sample(image, label, options, rng)
            original, normalized, truth = _to_tensors(crop, label_crop, device)
            keep_fraction = (max(1e-8, min(1 - 1e-8, rng.random())) if official_phase3 else
                             fractions[(epoch * len(pairs) + index) % len(fractions)])
            with autocast():
                latent = core.encoder(normalized)
                saliency = core.sal_decoder(latent)
                retained = (_official_phase3_keep_mask(saliency, keep_fraction)
                            if official_phase3 else priority_mask(saliency, keep_fraction))
                transmitted = latent * retained
                if official_phase3:
                    # 官方模型第三阶段在保留的潜变量上加入 U(-0.01,0.01) 噪声。
                    noise = torch.rand_like(transmitted, dtype=torch.float32) * 0.02 - 0.01
                    transmitted = transmitted + noise.to(transmitted.dtype)
                reconstruction = core.decoder(transmitted)
                # 固定的 SegFormer-B5 参数允许梯度穿过输入回传至 LimitNet。
                logits = segmenter(pixel_values=(reconstruction - seg_mean) / seg_std).logits
                logits = F.interpolate(logits.float(), size=truth.shape[-2:],
                                       mode="bilinear", align_corners=False)
                task_loss = F.cross_entropy(logits, truth, ignore_index=255)
                reconstruction_loss = F.l1_loss(reconstruction.float(), original)
                if official_phase3:
                    saliency_loss = task_loss.new_zeros(())
                    total = task_loss
                else:
                    focus = torch.zeros_like(truth, dtype=torch.float32)
                    for class_id in TARGET_IDS:
                        focus = torch.maximum(focus, (truth == class_id).float())
                    focus = F.interpolate(focus.unsqueeze(1), size=saliency.shape[-2:],
                                          mode="area")
                    # Sigmoid 后的 BCE 需在自动混合精度外计算，避免不安全的反向传播。
                    with torch.autocast("cuda", enabled=False) if bf16 else nullcontext():
                        saliency_loss = F.binary_cross_entropy(saliency.float(), focus)
                    total = (options.task_weight * task_loss +
                             options.reconstruction_weight * reconstruction_loss +
                             options.saliency_weight * saliency_loss)
            if not torch.isfinite(total):
                raise RuntimeError(f"LimitNet 损失出现非有限值：{image_path}")
            block_start = (index // options.accumulation) * options.accumulation
            block_size = min(options.accumulation, len(pairs) - block_start)
            (total / block_size).backward()
            for name, value in (("task", task_loss), ("reconstruction", reconstruction_loss),
                                ("saliency", saliency_loss), ("total", total)):
                running[name] += float(value.detach().cpu())
            if (index + 1) % options.accumulation == 0 or index + 1 == len(pairs):
                nn.utils.clip_grad_norm_(core.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        averages = {name: value / len(pairs) for name, value in running.items()}
        history.append({"epoch": epoch + 1, "images": len(pairs), "loss": averages})
        print(f"LimitNet epoch {epoch + 1}/{options.epochs}: {averages}", flush=True)
        options.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": source, "architecture": "official_limitnet_conv_encoder_saldecoder_decoder",
            "training_dataset": "Cityscapes train", "training_images_per_epoch": len(pairs),
            "crop_size": options.crop_size if options.input_mode == "native_crop" else None,
            "input_mode": options.input_mode,
            "codec_input_resolution": ([224, 224] if options.input_mode ==
                                       "official224_full_frame" else None),
            "pixel_rescaling": options.input_mode == "official224_full_frame",
            "task_model": str(options.segmenter_path), "target_class_ids": list(TARGET_IDS),
            "training_recipe": options.training_recipe,
            "keep_fractions": None if official_phase3 else list(fractions),
            "keep_fraction_policy": ("uniform_0_1_per_image" if official_phase3 else
                                     "cyclic_0.2_0.4_0.6_0.8_1.0"),
            "saliency_branch_frozen": official_phase3,
            "latent_noise_uniform": [-0.01, 0.01] if official_phase3 else None,
            "optimizer": "Adam" if official_phase3 else "AdamW",
            "epochs_completed": epoch + 1,
            "learning_rate": effective_lr, "accumulation": options.accumulation,
            "loss_weights": ({"task_ce": 1.0, "reconstruction_l1": 0.0,
                              "saliency_bce": 0.0} if official_phase3 else
                             {"task_ce": options.task_weight,
                              "reconstruction_l1": options.reconstruction_weight,
                              "saliency_bce": options.saliency_weight}),
            "precision": "bf16" if bf16 else "fp32", "seed": options.seed,
            "init_checkpoint": str(options.init_checkpoint) if options.init_checkpoint else None,
        }
        cpu_state = {key: value.detach().cpu() for key, value in core.state_dict().items()}
        torch.save({"state_dict": cpu_state, "metadata": metadata},
                   options.output_checkpoint)
    return {"checkpoint": str(options.output_checkpoint), "metadata": metadata,
            "history": history}
