"""统一读取项目配置，避免各模块自行解释相对路径。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ReplayConfig:
    root: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values[name]
        if not isinstance(value, dict):
            raise TypeError(f"配置节 {name} 必须是映射")
        return value

    def path(self, value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root / candidate

    def model_path(self, name: str) -> Path:
        return self.path(self.section("models")[name]["path"])

    def data_path(self, name: str) -> Path:
        return self.path(self.section("data")[name])

    def receiver_checkpoint(self) -> Path:
        base = self.data_path("checkpoint")
        projection = self.section("models")["clip"]["patch_projection"]
        return base.with_name(f"{base.stem}_{projection}{base.suffix}")


def load_config(path: str | Path) -> ReplayConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("配置文件必须是 YAML 映射")
    # 配置可以从其他目录调用，路径始终相对于 replay 项目根目录。
    root = next((parent for parent in path.parents if (parent / "pyproject.toml").exists()), None)
    if root is None:
        raise ValueError(f"未找到项目根目录：{path}")
    return ReplayConfig(root=root, values=values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 没有可用的 NVIDIA GPU")
    return torch.device(name)
