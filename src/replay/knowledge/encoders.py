"""固定的 CLIP 图文编码器，即图像模态的 SharedKB。"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


class SharedKB:
    def __init__(self, model_path: Path, device: torch.device,
                 patch_projection: str = "post_layernorm"):
        if patch_projection not in {"post_layernorm", "raw_hidden"}:
            raise ValueError("patch_projection 必须是 post_layernorm 或 raw_hidden")
        self.device = device
        self.patch_projection = patch_projection
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval()

    @classmethod
    def from_config(cls, config, device: torch.device) -> "SharedKB":
        from replay.core.assets import verify_model_lock
        verify_model_lock(config, "clip")
        return cls(config.model_path("clip"), device,
                   patch_projection=config.section("models")["clip"]["patch_projection"])

    @staticmethod
    def _normalized(tensor: torch.Tensor) -> np.ndarray:
        tensor = torch.nn.functional.normalize(tensor.float(), dim=-1)
        return tensor.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("文本列表不能为空")
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        features = self.model.get_text_features(**{key: value.to(self.device) for key, value in inputs.items()})
        return self._normalized(features)

    @torch.inference_mode()
    def images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("图像列表不能为空")
        # CLIP 自身要求固定视觉输入；这里只缩放特征提取副本，原图仍用于传输与恢复。
        inputs = self.processor(images=[image.convert("RGB") for image in images], return_tensors="pt")
        features = self.model.get_image_features(**{key: value.to(self.device) for key, value in inputs.items()})
        return self._normalized(features)

    @torch.inference_mode()
    def image_tokens(self, image: Image.Image, batch_tiles: int = 16) -> np.ndarray:
        """保留每个原图 16×16 patch 的位置；边缘只补齐，不缩小整图。"""
        image = image.convert("RGB")
        patch = int(self.model.config.vision_config.patch_size)
        side = int(self.model.config.vision_config.image_size)
        if side % patch:
            raise ValueError("CLIP 图像输入边长必须是 patch 的整数倍")
        tile_patches = side // patch
        rows = math.ceil(image.height / patch)
        cols = math.ceil(image.width / patch)
        result = np.empty((rows, cols, int(self.model.config.projection_dim)), dtype=np.float32)
        tiles: list[Image.Image] = []
        positions: list[tuple[int, int]] = []
        for y in range(0, image.height, side):
            for x in range(0, image.width, side):
                crop = image.crop((x, y, min(x + side, image.width), min(y + side, image.height)))
                tile = Image.new("RGB", (side, side))
                tile.paste(crop, (0, 0))
                tiles.append(tile)
                positions.append((y, x))
        if batch_tiles < 1:
            raise ValueError("batch_tiles 必须为正")
        for start in range(0, len(tiles), batch_tiles):
            inputs = self.processor(images=tiles[start:start + batch_tiles], return_tensors="pt",
                                    do_resize=False, do_center_crop=False)
            output = self.model.vision_model(pixel_values=inputs["pixel_values"].to(self.device))
            # CLIP 公开权重只直接训练了池化特征的对齐；逐 patch 投影是论文未公开的细节。
            hidden = output.last_hidden_state[:, 1:, :]
            if self.patch_projection == "post_layernorm":
                hidden = self.model.vision_model.post_layernorm(hidden)
            tokens = self.model.visual_projection(hidden)
            token_batch = self._normalized(tokens).reshape(-1, tile_patches, tile_patches, result.shape[-1])
            for (y, x), token_map in zip(positions[start:start + batch_tiles], token_batch):
                keep_y = min(tile_patches, rows - y // patch)
                keep_x = min(tile_patches, cols - x // patch)
                result[y // patch:y // patch + keep_y, x // patch:x // patch + keep_x] = token_map[:keep_y, :keep_x]
        return result
