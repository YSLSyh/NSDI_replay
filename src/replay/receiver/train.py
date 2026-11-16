"""使用当前清单中的 COCO 四折与 FSS-1000 原图做探索性接收端预训练。"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from replay.core.config import ReplayConfig, seed_everything, write_json
from replay.data.manifest import read_jsonl
from replay.data.token_cache import token_path, verify_cache
from replay.knowledge.encoders import SharedKB
from replay.knowledge.store import PosteriorKB
from replay.receiver.corruption import drop_semantic_tokens
from replay.receiver.network import FiLMUNet


class RestorationDataset(Dataset):
    def __init__(self, manifest: Path, cache_dir: Path):
        self.rows = read_jsonl(manifest)
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path"]) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        cached = token_path(self.cache_dir, row)
        if not cached.exists():
            raise FileNotFoundError(f"缺少 FP32 token 缓存：{cached}；先运行 replay data cache-tokens")
        tokens = torch.from_numpy(np.load(cached).copy())
        return tensor, tokens, row["bins"][0], f"{row['source']}:{row['source_id']}:"


def _context(current_tokens: torch.Tensor, label: str, sample_id: str,
             shared: SharedKB, kb: PosteriorKB, k: int, device: torch.device,
             text_cache: dict[str, np.ndarray]) -> torch.Tensor:
    # 同一类别的文本特征固定，跨 epoch 复用，避免反复运行 CLIP 文本塔。
    if label not in text_cache:
        text_cache[label] = shared.text([label])[0]
    query = text_cache[label]
    reference = kb.context(query, k=k, exclude_id=sample_id)
    return torch.cat([current_tokens, torch.from_numpy(reference)[None].to(device)], dim=-1)


def train_receiver(config: ReplayConfig, device: torch.device, shared: SharedKB, kb: PosteriorKB) -> dict:
    seed_everything(int(config.values["seed"]))
    spec = config.section("receiver")
    if int(spec["batch_size"]) != 1:
        raise ValueError("原始尺寸各异，当前训练器固定 batch_size=1")
    cache_dir = verify_cache(config)
    train = RestorationDataset(config.data_path("train_manifest"), cache_dir)
    val = RestorationDataset(config.data_path("val_manifest"), cache_dir)
    if not len(train) or not len(val):
        raise ValueError("训练或验证清单为空")
    loaders = {
        "train": DataLoader(train, batch_size=1, shuffle=True, num_workers=int(spec["num_workers"])),
        "val": DataLoader(val, batch_size=1, shuffle=False, num_workers=int(spec["num_workers"])),
    }
    network = FiLMUNet(int(spec["base_channels"]), int(spec["context_dim"])).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(spec["adam_lr"]),
                                 betas=(float(spec["adam_beta1"]), float(spec["adam_beta2"])))
    epochs = int(spec["epochs"])
    initial, minimum = float(spec["adam_lr"]), float(spec["min_lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (minimum + (initial - minimum) * (1 + math.cos(math.pi * epoch / epochs)) / 2) / initial)
    checkpoint = config.receiver_checkpoint()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    patch = int(config.section("image")["patch_size"])
    top_k = int(config.section("knowledge")["rx_top_k"])
    text_cache: dict[str, np.ndarray] = {}
    for epoch in range(epochs):
        losses = {}
        for split in ("train", "val"):
            network.train(split == "train")
            total = 0.0
            for original, token_grid, labels, sample_ids in tqdm(loaders[split], desc=f"epoch {epoch + 1} {split}"):
                original = original.to(device)
                generator = None
                if split == "val":
                    # 每张验证图的丢失模式固定，跨 epoch 和投影候选保持同一随机条件。
                    digest = hashlib.sha256(sample_ids[0].encode("utf-8")).digest()
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(config.values["seed"]) + int.from_bytes(digest[:8], "big"))
                damaged, missing, current = drop_semantic_tokens(
                    original, token_grid, patch, float(spec["dropout_min"]),
                    float(spec["dropout_max"]), generator=generator)
                context = _context(current, labels[0], sample_ids[0], shared, kb,
                                   top_k, device, text_cache)
                with torch.set_grad_enabled(split == "train"):
                    output = network(damaged, missing, context)
                    loss = F.l1_loss(output, original)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach())
            losses[split] = total / len(loaders[split])
        history.append({"epoch": epoch + 1, "train_l1": losses["train"],
                        "val_l1": losses["val"], "lr": optimizer.param_groups[0]["lr"]})
        if losses["val"] < best:
            best = losses["val"]
            torch.save({"state_dict": network.state_dict(), "epoch": epoch + 1,
                        "val_l1": best, "receiver": dict(spec),
                        "clip_patch_projection": config.section("models")["clip"]["patch_projection"]}, checkpoint)
        scheduler.step()
        projection = config.section("models")["clip"]["patch_projection"]
        write_json(config.root / "artifacts" / f"receiver_history_{projection}.json", history)
    return {"checkpoint": str(checkpoint), "best_val_l1": best,
            "parameters": sum(p.numel() for p in network.parameters())}
