"""四卡 BF16 连续训练 15 轮；逐轮保存 Adam 状态，只在末轮完整验证。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from replay.core.config import load_config, seed_everything, write_json
from replay.data.token_cache import verify_cache
from replay.knowledge.encoders import SharedKB
from replay.knowledge.store import PosteriorKB
from replay.receiver.corruption import drop_semantic_tokens
from replay.receiver.network import FiLMUNet
from replay.receiver.train import RestorationDataset
from train_pilot_ddp import ExactSizeBatchSampler, cached_context, reduce_loss


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_fold0_bf16_15_epochs.yaml")
    parser.add_argument("--result-path", default="artifacts/pilot_fold0_bf16_15ep_result.json")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    spec = config.section("receiver")
    epochs = int(spec["epochs"])
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 4 or int(spec["world_size"]) != 4 or epochs != 15:
        raise ValueError("本入口只用于四卡 BF16 连续 15 轮")
    if spec["precision"] != "bfloat16":
        raise ValueError("本入口要求 receiver.precision=bfloat16")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {local_rank} 不支持 BF16")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl")
    try:
        seed = int(config.values["seed"])
        seed_everything(seed + rank)
        cache_dir = verify_cache(config)
        train = RestorationDataset(config.data_path("train_manifest"), cache_dir)
        val = RestorationDataset(config.data_path("val_manifest"), cache_dir)
        batch_size = int(spec["batch_size"])
        rules = spec["area_batch_rules"]
        train_sampler = ExactSizeBatchSampler(train.rows, batch_size, rank, world, seed,
                                             training=True, area_batch_rules=rules)
        val_sampler = ExactSizeBatchSampler(val.rows, batch_size, rank, world, seed,
                                           training=False, area_batch_rules=rules)
        workers = int(spec["num_workers"])
        train_loader = DataLoader(train, batch_sampler=train_sampler, num_workers=workers,
                                  pin_memory=True, persistent_workers=workers > 0)
        val_loader = DataLoader(val, batch_sampler=val_sampler, num_workers=workers,
                                pin_memory=True, persistent_workers=workers > 0)
        shared = SharedKB.from_config(config, device)
        kb = PosteriorKB.load(config.data_path("kb_path"))
        network = FiLMUNet(int(spec["base_channels"]), int(spec["context_dim"])).to(device)
        model = DistributedDataParallel(network, device_ids=[local_rank],
                                        output_device=local_rank, broadcast_buffers=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["adam_lr"]),
                                     betas=(float(spec["adam_beta1"]), float(spec["adam_beta2"])))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(spec["min_lr"]))
        checkpoint_path = config.receiver_checkpoint()
        start_epoch = 0
        history: list[dict] = []
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if saved.get("precision") != "bfloat16" or saved.get("world_size") != world:
                raise ValueError("已有权重的精度或 GPU 数与本次训练不一致")
            network.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start_epoch = int(saved["epoch"])
            history = list(saved["history"])
            if rank == 0:
                print(f"从第 {start_epoch + 1} 轮继续训练", flush=True)
        patch = int(config.section("image")["patch_size"])
        top_k = int(config.section("knowledge")["rx_top_k"])
        retrieval_cache: dict[str, list[dict]] = {}

        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            lr_used = optimizer.param_groups[0]["lr"]
            epoch_start = time.perf_counter()
            loss_sum, image_count = 0.0, 0
            iterator = tqdm(train_loader, desc=f"BF16 epoch {epoch + 1}/{epochs} train",
                            disable=rank != 0, mininterval=2)
            for original, token_grid, labels, sample_ids in iterator:
                original = original.to(device, non_blocking=True)
                damaged, missing, current = drop_semantic_tokens(
                    original, token_grid, patch, float(spec["dropout_min"]),
                    float(spec["dropout_max"]))
                context = torch.cat([
                    cached_context(current[i:i + 1], labels[i], sample_ids[i], shared, kb,
                                   top_k, device, retrieval_cache)
                    for i in range(original.shape[0])
                ])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(damaged, missing, context)
                loss = F.l1_loss(output.float(), original)
                global_count = torch.tensor([original.shape[0]], dtype=torch.float32, device=device)
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
                (loss * (world * original.shape[0] / global_count.item())).backward()
                optimizer.step()
                count = original.shape[0]
                loss_sum += float(loss.detach()) * count
                image_count += count
            train_l1 = reduce_loss(loss_sum, image_count, device)
            train_seconds = time.perf_counter() - epoch_start
            scheduler.step()
            row = {"epoch": epoch + 1, "train_l1": train_l1, "lr": lr_used,
                   "train_seconds": train_seconds}

            if epoch + 1 == epochs:
                model.eval()
                val_start = time.perf_counter()
                val_sum, val_count = 0.0, 0
                iterator = tqdm(val_loader, desc="BF16 final validation",
                                disable=rank != 0, mininterval=2)
                with torch.inference_mode():
                    for original, token_grid, labels, sample_ids in iterator:
                        original = original.to(device, non_blocking=True)
                        damaged_list, missing_list, current_list = [], [], []
                        for image, tokens, sample_id in zip(original, token_grid, sample_ids):
                            digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
                            generator = torch.Generator(device=device)
                            generator.manual_seed((seed + int.from_bytes(digest[:8], "big")) % (2**63))
                            damaged, missing, current = drop_semantic_tokens(
                                image[None], tokens[None], patch,
                                float(spec["dropout_min"]), float(spec["dropout_max"]), generator)
                            damaged_list.append(damaged)
                            missing_list.append(missing)
                            current_list.append(current)
                        damaged = torch.cat(damaged_list)
                        missing = torch.cat(missing_list)
                        current = torch.cat(current_list)
                        context = torch.cat([
                            cached_context(current[i:i + 1], labels[i], sample_ids[i], shared, kb,
                                           top_k, device, retrieval_cache)
                            for i in range(original.shape[0])
                        ])
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = network(damaged, missing, context)
                        loss = F.l1_loss(output.float(), original)
                        count = original.shape[0]
                        val_sum += float(loss) * count
                        val_count += count
                row["val_l1"] = reduce_loss(val_sum, val_count, device)
                row["val_seconds"] = time.perf_counter() - val_start

            history.append(row)
            if rank == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"state_dict": network.state_dict(),
                           "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(),
                           "epoch": epoch + 1, "history": history,
                           "precision": "bfloat16", "world_size": world,
                           "receiver": dict(spec), "fold": 0}
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
                torch.save(payload, temporary)
                temporary.replace(checkpoint_path)
                result = {"checkpoint": str(checkpoint_path), "completed_epochs": epoch + 1,
                          "target_epochs": epochs, "train_images": len(train),
                          "val_images": len(val), "precision": "bfloat16 autocast",
                          "world_size": world, "area_batch_rules": rules,
                          "history": history,
                          "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                write_json(ROOT / args.result_path, result)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            dist.barrier()
    finally:
        # 验证可能留下大量 CUDA 缓存；先释放，避免 NCCL 在退出清理时分配失败。
        torch.cuda.empty_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
