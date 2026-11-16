"""在远程服务器上下载并固定模型修订版本。"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from replay.core.config import ReplayConfig, write_json


ALLOW = {name: ["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"]
         for name in ("clip", "clipseg", "qwen", "segmenter")}


def model_lock_path(config: ReplayConfig) -> Path:
    """允许实验配置使用独立模型锁，避免改动已有的 0.6B 基线。"""
    return config.path(config.values.get("models_lock_file", "artifacts/models.lock.json"))


def download_models(config: ReplayConfig, names: list[str]) -> dict[str, dict[str, str]]:
    api = HfApi()
    lock_path = model_lock_path(config)
    lock: dict[str, dict[str, str]] = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    for name in names:
        if name not in ALLOW:
            raise ValueError(f"未知模型名称：{name}")
        spec = config.section("models")[name]
        # main 这样的浮动引用必须先解析成提交哈希，再写入锁文件。
        revision = str(spec["revision"])
        sha = api.model_info(spec["repo"], revision=revision).sha
        destination: Path = config.model_path(name)
        snapshot_download(
            repo_id=spec["repo"],
            revision=sha,
            local_dir=destination,
            allow_patterns=ALLOW[name],
        )
        lock[name] = {"repo": spec["repo"], "revision": sha, "path": str(destination)}
    write_json(lock_path, lock)
    return lock


def verify_model_lock(config: ReplayConfig, name: str) -> None:
    lock_path = model_lock_path(config)
    if not lock_path.exists():
        raise FileNotFoundError("缺少模型锁文件；请先运行 python scripts/download_models.py")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if name not in lock:
        raise ValueError(f"锁文件尚未记录模型 {name}")
    expected = config.section("models")[name]
    entry = lock[name]
    if entry["repo"] != expected["repo"] or (
        expected["revision"] != "main" and entry["revision"] != expected["revision"]
    ) or Path(entry["path"]).resolve() != config.model_path(name).resolve():
        raise ValueError(f"模型 {name} 的实际锁文件与配置不一致")
    if not config.model_path(name).exists():
        raise FileNotFoundError(f"模型目录缺失：{config.model_path(name)}")
