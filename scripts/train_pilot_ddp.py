"""第 0 折接收端一轮 DDP 训练；保留原图尺寸，可选 FP32 或 BF16 计算。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from replay.core.config import load_config, seed_everything, write_json
from replay.data.token_cache import verify_cache
from replay.knowledge.encoders import SharedKB
from replay.knowledge.store import PosteriorKB
from replay.receiver.corruption import drop_semantic_tokens
from replay.receiver.network import FiLMUNet
from replay.receiver.train import RestorationDataset


ROOT = Path(__file__).resolve().parents[1]


class ExactSizeBatchSampler(Sampler[list[int]]):
    """同尺寸原图组批；大图自动降到单卡 1 张，所有原图都保留。"""

    def __init__(self, rows: list[dict], batch_size: int, rank: int, world: int,
                 seed: int, training: bool, area_batch_rules: list[dict] | None = None):
        if batch_size < 1:
            raise ValueError("batch_size 必须大于零")
        self.batch_size, self.rank, self.world = batch_size, rank, world
        self.seed, self.training = seed, training
        # 默认规则保持此前三卡 FP32 实验的组批行为；BF16 规则由实测后写入独立配置。
        self.area_batch_rules = area_batch_rules or [{"max_pixels": 307200, "batch_size": 2}]
        self.area_batch_rules = sorted(self.area_batch_rules, key=lambda item: int(item["max_pixels"]))
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[(int(row["height"]), int(row["width"]))].append(index)
        self.groups = groups
        self.epoch = 0
        self.duplicate_images = 0
        count = sum((len(indices) + self.size_for(shape) - 1) // self.size_for(shape)
                    for shape, indices in groups.items())
        self.duplicate_batches = (-count) % world if training else 0
        self.total_batches = count + self.duplicate_batches

    def size_for(self, shape: tuple[int, int]) -> int:
        height, width = shape
        for rule in self.area_batch_rules:
            if height * width <= int(rule["max_pixels"]):
                return min(self.batch_size, int(rule["batch_size"]))
        return 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for shape, indices in self.groups.items():
            items = list(indices)
            if self.training:
                rng.shuffle(items)
            group_size = self.size_for(shape)
            for offset in range(0, len(items), group_size):
                batches.append(items[offset:offset + group_size])
        if self.training:
            rng.shuffle(batches)
            batches.extend(batches[:self.duplicate_batches])
        return iter(batches[self.rank::self.world])

    def __len__(self) -> int:
        return len(range(self.rank, self.total_batches, self.world))


def reduce_loss(sum_loss: float, count: int, device: torch.device) -> float:
    values = torch.tensor([sum_loss, float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0] / values[1])


def cached_context(current_tokens: torch.Tensor, label: str, sample_id: str,
                   shared: SharedKB, kb: PosteriorKB, k: int, device: torch.device,
                   retrieval_cache: dict[str, list[dict]]) -> torch.Tensor:
    """同一类别只检索一次；仍逐样本排除其自身，防止知识库直接泄漏答案。"""
    if label not in retrieval_cache:
        query = shared.text([label])[0]
        # 额外取一条，排除当前图片后仍有 k 条可用参考。
        retrieval_cache[label] = kb.search(query, k=k + 1, require_image=True)
    matches = [hit["data_vector"] for hit in retrieval_cache[label]
               if not str(hit["record"].get("id", "")).startswith(sample_id)][:k]
    if matches:
        vector = np.mean(matches, axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
    else:
        vector = np.zeros(kb.dim, dtype=np.float32)
    return torch.cat([current_tokens, torch.from_numpy(vector)[None].to(device)], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_fold0_one_epoch.yaml")
    parser.add_argument("--result-path", default="artifacts/pilot_fold0_train_result.json")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    config = load_config(ROOT / args.config)
    spec = config.section("receiver")
    if world != int(spec.get("world_size", 3)):
        raise ValueError("torchrun 进程数与 receiver.world_size 不一致")
    precision = str(spec.get("precision", "float32"))
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("receiver.precision 仅支持 float32 或 bfloat16")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {local_rank} 不支持 BF16")
    # TF32 始终关闭；BF16 只用于接收端前后向计算，权重/Adam/CLIP 缓存保持 FP32。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(backend="nccl")
    try:
        seed = int(config.values["seed"])
        seed_everything(seed)
        if int(spec["epochs"]) != 1:
            raise ValueError("短训练只运行一轮")
        batch_size = int(spec["batch_size"])
        area_batch_rules = spec.get("area_batch_rules")
        cache_dir = verify_cache(config)
        train = RestorationDataset(config.data_path("train_manifest"), cache_dir)
        val = RestorationDataset(config.data_path("val_manifest"), cache_dir)
        train_sampler = ExactSizeBatchSampler(train.rows, batch_size, rank, world, seed,
                                             training=True, area_batch_rules=area_batch_rules)
        val_sampler = ExactSizeBatchSampler(val.rows, batch_size, rank, world, seed,
                                           training=False, area_batch_rules=area_batch_rules)
        workers = int(spec["num_workers"])
        loaders = {
            "train": DataLoader(train, batch_sampler=train_sampler, num_workers=workers,
                                pin_memory=True, persistent_workers=workers > 0),
            "val": DataLoader(val, batch_sampler=val_sampler, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0),
        }
        shared = SharedKB.from_config(config, device)
        kb = PosteriorKB.load(config.data_path("kb_path"))
        network = FiLMUNet(int(spec["base_channels"]), int(spec["context_dim"])).to(device)
        model = DistributedDataParallel(network, device_ids=[local_rank],
                                        output_device=local_rank, broadcast_buffers=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["adam_lr"]),
                                     betas=(float(spec["adam_beta1"]), float(spec["adam_beta2"])))
        patch = int(config.section("image")["patch_size"])
        top_k = int(config.section("knowledge")["rx_top_k"])
        retrieval_cache: dict[str, list[dict]] = {}
        metrics = {}
        for split in ("train", "val"):
            model.train(split == "train")
            loss_sum, image_count = 0.0, 0
            iterator = tqdm(loaders[split], desc=f"pilot fold0 {split}", disable=rank != 0)
            for original, token_grid, labels, sample_ids in iterator:
                original = original.to(device, non_blocking=True)
                generator = None
                if split == "val":
                    # 同一验证样本每次使用固定缺失模式；批内按样本分别生成随机数。
                    damaged_list, missing_list, current_list = [], [], []
                    for image, tokens, sample_id in zip(original, token_grid, sample_ids):
                        digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
                        generator = torch.Generator(device=device)
                        generator.manual_seed((seed + int.from_bytes(digest[:8], "big")) % (2**63))
                        d, m, c = drop_semantic_tokens(image[None], tokens[None], patch,
                                                       float(spec["dropout_min"]),
                                                       float(spec["dropout_max"]), generator)
                        damaged_list.append(d)
                        missing_list.append(m)
                        current_list.append(c)
                    damaged = torch.cat(damaged_list)
                    missing = torch.cat(missing_list)
                    current = torch.cat(current_list)
                else:
                    damaged, missing, current = drop_semantic_tokens(
                        original, token_grid, patch, float(spec["dropout_min"]),
                        float(spec["dropout_max"]))
                context = torch.cat([
                    cached_context(current[i:i + 1], labels[i], sample_ids[i], shared, kb,
                                   top_k, device, retrieval_cache)
                    for i in range(original.shape[0])
                ])
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=precision == "bfloat16"):
                        output = model(damaged, missing, context)
                    # L1 与梯度标量仍用 FP32，BF16 不需 GradScaler。
                    loss = F.l1_loss(output.float(), original)
                    # 各卡本步可能分别有 1 或 2 张图；按真实图像数加权后再同步梯度。
                    global_count = torch.tensor([original.shape[0]], dtype=torch.float32, device=device)
                    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
                    (loss * (world * original.shape[0] / global_count.item())).backward()
                    optimizer.step()
                else:
                    # 验证样本数在各卡可不同，绕开 DDP 的逐步同步后汇总精确指标。
                    with torch.inference_mode():
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=precision == "bfloat16"):
                            output = network(damaged, missing, context)
                        loss = F.l1_loss(output.float(), original)
                count = original.shape[0]
                loss_sum += float(loss.detach()) * count
                image_count += count
            metrics[f"{split}_l1"] = reduce_loss(loss_sum, image_count, device)
            if rank == 0:
                print(json.dumps({"split": split, "l1": metrics[f"{split}_l1"],
                                  "rank0_processed_images": image_count},
                                 ensure_ascii=False), flush=True)
        dist.barrier()
        if rank == 0:
            checkpoint = config.receiver_checkpoint()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": network.state_dict(), "epoch": 1,
                        "val_l1": metrics["val_l1"], "receiver": dict(spec),
                        "clip_patch_projection": config.section("models")["clip"]["patch_projection"],
                        "world_size": world, "per_gpu_batch_size": batch_size,
                        "precision": precision, "fold": 0}, checkpoint)
            result = {"checkpoint": str(checkpoint), "epoch": 1,
                      "train_l1": metrics["train_l1"], "val_l1": metrics["val_l1"],
                      "train_images": len(train), "val_images": len(val),
                      "per_gpu_batch_size": batch_size, "world_size": world,
                      "global_batch_size_max": batch_size * world,
                      "area_batch_rules": train_sampler.area_batch_rules,
                      "precision": precision,
                      "duplicate_images_for_exact_size_batches": train_sampler.duplicate_images,
                      "duplicate_batches_for_equal_steps": train_sampler.duplicate_batches,
                      "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
            write_json(ROOT / args.result_path, result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
