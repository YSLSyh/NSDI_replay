"""加载已训练接收端并用检索上下文修复传输图像。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch

from replay.core.config import ReplayConfig
from replay.receiver.network import FiLMUNet

if TYPE_CHECKING:
    from replay.knowledge.encoders import SharedKB
    from replay.knowledge.store import PosteriorKB


def load_receiver(config: ReplayConfig, device: torch.device,
                  *, checkpoint_path: str | Path | None = None) -> FiLMUNet:
    spec = config.section("receiver")
    network = FiLMUNet(int(spec["base_channels"]), int(spec["context_dim"])).to(device)
    path = (config.receiver_checkpoint() if checkpoint_path is None
            else config.path(checkpoint_path))
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    projection = config.section("models")["clip"]["patch_projection"]
    recorded_projection = checkpoint.get("clip_patch_projection")
    if (recorded_projection is None and
            path.name == "receiver_pilot_fold0_bf16_15ep_post_layernorm.pt" and
            projection == "post_layernorm" and checkpoint.get("epoch") == 15 and
            checkpoint.get("precision") == "bfloat16" and checkpoint.get("fold") == 0):
        # 15 轮训练入口把投影模式记在权重文件名，而没有写入 payload。
        recorded_projection = "post_layernorm"
    if recorded_projection != projection:
        raise ValueError("接收端权重与 CLIP patch 投影选择不一致")
    network.load_state_dict(checkpoint["state_dict"])
    return network.eval()


@torch.inference_mode()
def restore(image: Image.Image, bins: list[str], shared: SharedKB,
            kb: PosteriorKB, model: FiLMUNet, top_k: int, device: torch.device,
            update_session: bool = True, merge_cosine: float = 0.95,
            missing_pixels: np.ndarray | None = None) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = None
    if missing_pixels is not None:
        mask = np.asarray(missing_pixels, dtype=np.float32)
        if mask.shape != rgb.shape[:2] or not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError("接收端缺失图必须与图像同尺寸且位于 [0,1]")
        # 训练时缺失区域为零值；码流丢包后的 JPEG 解码器可能把它显示成白带。
        # 仅在恢复网络输入中归零，直接解码图仍保留真实的解码结果用于对照。
        rgb = np.rint(rgb.astype(np.float32) * (1.0 - mask[..., None])).astype(np.uint8)
    receiver_input = Image.fromarray(rgb)
    text = shared.text([", ".join(bins)])[0]
    tokens = shared.image_tokens(receiver_input)
    if mask is None:
        current = tokens.mean(axis=(0, 1))
    else:
        patch = int(shared.model.config.vision_config.patch_size)
        rows, cols = tokens.shape[:2]
        padded = np.pad(mask, ((0, rows * patch - mask.shape[0]),
                               (0, cols * patch - mask.shape[1])), mode="constant")
        lost = padded.reshape(rows, patch, cols, patch).mean(axis=(1, 3)) >= 0.5
        # 与预训练一致：上下文只平均幸存的语义 token。
        current = tokens[~lost].mean(axis=0) if (~lost).any() else np.zeros(tokens.shape[-1], np.float32)
    current = current / max(float(np.linalg.norm(current)), 1e-12)
    # 当前图像的幸存 token 摘要与知识库检索的参考上下文拼接，作为 FiLM 条件。
    reference = kb.context(text, k=top_k)
    context = torch.from_numpy(np.concatenate([current, reference])[None]).to(device)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device).float() / 255.0
    if mask is None:
        missing = torch.zeros_like(tensor[:, :1])
    else:
        # 缺失区域已按训练分布归零，mask 另行提供给网络定位缺口。
        missing = torch.from_numpy(mask.copy())[None, None].to(device)
    result = model(tensor, missing, context)[0].permute(1, 2, 0).cpu().numpy()
    restored = Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))
    if update_session:
        kb.insert_session(text, shared.images([restored])[0], bins, merge_cosine)
    return restored
