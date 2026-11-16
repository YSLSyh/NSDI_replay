"""论文 Algorithm 1/2 的会话语义 B 与逐帧重要性图 M。"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np


# 1 MiB 位掩码已覆盖八百多万个 patch，限制尺寸也防止压缩数据膨胀。
_MAX_FRAME_MASK_BYTES = 1 << 20
_MAX_FRAME_HEADER_BYTES = 2 << 20


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class SessionHeader:
    task: str
    bins: tuple[str, ...]
    shared_model_repo: str
    shared_model_revision: str

    def to_bytes(self) -> bytes:
        # 同一会话只发送一次；接收端据此确认 SharedKB 版本一致。
        return _json_bytes({"schema": 1, "task": self.task, "bins": list(self.bins),
                            "shared_model_repo": self.shared_model_repo,
                            "shared_model_revision": self.shared_model_revision})

    @classmethod
    def from_bytes(cls, payload: bytes) -> "SessionHeader":
        value = json.loads(payload.decode("utf-8"))
        if value.get("schema") != 1 or not value.get("bins"):
            raise ValueError("不支持或为空的会话语义 header")
        return cls(task=value["task"], bins=tuple(value["bins"]),
                   shared_model_repo=value["shared_model_repo"],
                   shared_model_revision=value["shared_model_revision"])

    def verify_model(self, repo: str, revision: str) -> None:
        if (self.shared_model_repo, self.shared_model_revision) != (repo, revision):
            raise ValueError("Tx/Rx SharedKB 模型版本不一致，应退回普通 codec 通路")


@dataclass(frozen=True)
class FrameHeader:
    width: int
    height: int
    patch_size: int
    selected: np.ndarray

    def to_bytes(self, *, compact: bool = False) -> bytes:
        if self.width <= 0 or self.height <= 0 or self.patch_size <= 0:
            raise ValueError("逐帧 header 尺寸无效")
        rows, cols = self.selected.shape
        if (rows, cols) != ((self.height + self.patch_size - 1) // self.patch_size,
                            (self.width + self.patch_size - 1) // self.patch_size):
            raise ValueError("逐帧 mask 与图像尺寸不匹配")
        packed = np.packbits(self.selected.reshape(-1).astype(np.uint8), bitorder="big").tobytes()
        if len(packed) > _MAX_FRAME_MASK_BYTES:
            raise ValueError("逐帧 mask 超过安全尺寸")
        fields = {"width": self.width, "height": self.height,
                  "patch_size": self.patch_size}
        legacy = _json_bytes({"schema": 1, **fields,
                              "mask": base64.b64encode(packed).decode("ascii")})
        if not compact:
            return legacy
        # 压缩只改变重要性图的无损表示；稠密高熵 mask 若压不小，仍发送旧格式。
        compressed = _json_bytes({"schema": 2, **fields, "mask_codec": "zlib",
                                  "mask": base64.b64encode(zlib.compress(packed, 9)).decode("ascii")})
        return compressed if len(compressed) < len(legacy) else legacy

    @classmethod
    def from_bytes(cls, payload: bytes) -> "FrameHeader":
        if len(payload) > _MAX_FRAME_HEADER_BYTES:
            raise ValueError("逐帧 header 超过安全尺寸")
        value = json.loads(payload.decode("utf-8"))
        schema = value.get("schema")
        if schema not in (1, 2):
            raise ValueError("不支持的逐帧 header")
        width, height, patch = (int(value[key]) for key in ("width", "height", "patch_size"))
        if width <= 0 or height <= 0 or patch <= 0:
            raise ValueError("逐帧 header 尺寸无效")
        rows, cols = (height + patch - 1) // patch, (width + patch - 1) // patch
        expected_bytes = (rows * cols + 7) // 8
        if expected_bytes > _MAX_FRAME_MASK_BYTES:
            raise ValueError("逐帧 mask 超过安全尺寸")
        encoded = base64.b64decode(value["mask"], validate=True)
        if schema == 1:
            packed = encoded
        else:
            if value.get("mask_codec") != "zlib":
                raise ValueError("逐帧 mask 压缩格式不受支持")
            # 最多解压预期长度+1字节；截断、尾随数据和膨胀攻击均拒绝。
            try:
                decoder = zlib.decompressobj()
                packed = decoder.decompress(encoded, expected_bytes + 1)
            except zlib.error as error:
                raise ValueError("逐帧 mask 压缩数据损坏") from error
            if (len(packed) != expected_bytes or not decoder.eof or
                    decoder.unconsumed_tail or decoder.unused_data):
                raise ValueError("逐帧 mask 解压长度或结尾无效")
        if len(packed) != expected_bytes:
            raise ValueError("逐帧 mask 数据长度错误")
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
        return cls(width, height, patch, bits[:rows * cols].reshape(rows, cols).astype(bool))
