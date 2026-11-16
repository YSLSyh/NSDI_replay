"""PosteriorKB：文本向量索引 + 图像参考特征 + 可追溯元数据。"""

from __future__ import annotations

import json
from uuid import uuid4
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from replay.core.config import write_json


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(vector))
    if length == 0:
        raise ValueError("知识库不接受零向量")
    return vector / length


class PosteriorKB:
    """每条记录可有图像参考，也可只是供 RAG 使用的类别定义。"""

    def __init__(self, directory: Path, dim: int = 512):
        self.directory = directory
        self.dim = dim
        self.records: list[dict[str, Any]] = []
        self.text_index = faiss.IndexFlatIP(dim)
        self.data_vectors: list[np.ndarray | None] = []

    def add(self, text_vector: np.ndarray, record: dict[str, Any], data_vector: np.ndarray | None = None) -> None:
        if record.get("split") not in {"train", "session"}:
            raise ValueError("PosteriorKB 只允许训练数据或当前会话数据；禁止验证/测试集泄漏")
        text = _unit(text_vector)
        if text.size != self.dim:
            raise ValueError(f"文本嵌入维度应为 {self.dim}，实际为 {text.size}")
        data = None if data_vector is None else _unit(data_vector)
        if data is not None and data.size != self.dim:
            raise ValueError(f"图像嵌入维度应为 {self.dim}，实际为 {data.size}")
        self.text_index.add(text[None, :])
        stored = dict(record)
        stored["has_data"] = data is not None
        self.records.append(stored)
        self.data_vectors.append(data)

    def insert_session(self, text_vector: np.ndarray, data_vector: np.ndarray,
                       tags: list[str], merge_cosine: float = 0.95) -> str:
        """接收后更新 KB；相似会话项按均值形成质心，阈值为项目设计。"""
        if not -1 <= merge_cosine <= 1:
            raise ValueError("聚类余弦阈值无效")
        text, data = _unit(text_vector), _unit(data_vector)
        for index, record in enumerate(self.records):
            if record.get("split") != "session" or self.data_vectors[index] is None:
                continue
            old_text = self.text_index.reconstruct(index)
            old_data = self.data_vectors[index]
            if float(text @ old_text) >= merge_cosine and float(data @ old_data) >= merge_cosine:
                count = int(record.get("session_count", 1))
                new_text = _unit((old_text * count + text) / (count + 1))
                self.data_vectors[index] = _unit((old_data * count + data) / (count + 1))
                record["session_count"] = count + 1
                vectors = self.text_index.reconstruct_n(0, self.text_index.ntotal)
                vectors[index] = new_text
                self.text_index.reset()
                self.text_index.add(vectors.astype(np.float32))
                return str(record["id"])
        record_id = f"session:{uuid4()}"
        self.add(text, {"id": record_id, "split": "session", "source": "restored",
                        "text": ", ".join(tags), "tags": list(tags), "session_count": 1}, data)
        return record_id

    def search(self, text_vector: np.ndarray, k: int, require_image: bool = False,
               exclude_id: str | None = None) -> list[dict[str, Any]]:
        if k <= 0 or not self.records:
            return []
        query = _unit(text_vector)
        if query.size != self.dim:
            raise ValueError("查询向量维度与知识库不一致")
        # FAISS 先按文本相似度排序；需要图像参考时再过滤只有定义的记录。
        probe = min(len(self.records), max(k * 4, 32))
        while True:
            scores, ids = self.text_index.search(query[None, :], probe)
            results: list[dict[str, Any]] = []
            for score, row_id in zip(scores[0], ids[0]):
                if row_id < 0:
                    continue
                data = self.data_vectors[int(row_id)]
                if require_image and data is None:
                    continue
                if exclude_id is not None and str(self.records[int(row_id)].get("id", "")).startswith(exclude_id):
                    continue
                results.append({
                    "score": float(score),
                    "record": self.records[int(row_id)],
                    "data_vector": data,
                })
                if len(results) == k:
                    return results
            if probe == len(self.records):
                return results
            probe = min(len(self.records), probe * 2)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.text_index, str(self.directory / "text.faiss"))
        data = np.stack([vector if vector is not None else np.zeros(self.dim, np.float32)
                         for vector in self.data_vectors]) if self.records else np.empty((0, self.dim), np.float32)
        np.save(self.directory / "data_vectors.npy", data)
        write_json(self.directory / "records.json", self.records)
        write_json(self.directory / "meta.json", {"dim": self.dim, "count": len(self.records)})

    def save_session(self, destination: Path) -> int:
        """只保存当前会话增量，避免把整个训练 KB 复制一遍。"""
        ids = [i for i, record in enumerate(self.records) if record.get("split") == "session"]
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "records.json", [self.records[i] for i in ids])
        text = np.stack([self.text_index.reconstruct(i) for i in ids]) if ids else np.empty((0, self.dim), np.float32)
        data = np.stack([self.data_vectors[i] for i in ids]) if ids else np.empty((0, self.dim), np.float32)
        np.save(destination / "text_vectors.npy", text)
        np.save(destination / "data_vectors.npy", data)
        return len(ids)

    def load_session(self, source: Path) -> int:
        records = json.loads((source / "records.json").read_text(encoding="utf-8"))
        text = np.load(source / "text_vectors.npy")
        data = np.load(source / "data_vectors.npy")
        if len(records) != len(text) or len(records) != len(data):
            raise ValueError("会话知识库文件长度不一致")
        for index, record in enumerate(records):
            if record.get("split") != "session":
                raise ValueError("会话增量包含非会话记录")
            self.add(text[index], record, data[index])
        return len(records)

    @classmethod
    def load(cls, directory: Path) -> "PosteriorKB":
        metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        kb = cls(directory, dim=int(metadata["dim"]))
        kb.records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        kb.text_index = faiss.read_index(str(directory / "text.faiss"))
        vectors = np.load(directory / "data_vectors.npy")
        kb.data_vectors = [vectors[i] if record.get("has_data", bool(record.get("image_path"))) else None
                           for i, record in enumerate(kb.records)]
        if len(kb.records) != kb.text_index.ntotal or len(kb.records) != len(kb.data_vectors):
            raise ValueError("PosteriorKB 索引、元数据与图像特征数量不一致")
        return kb

    def context(self, text_vector: np.ndarray, k: int, exclude_id: str | None = None) -> np.ndarray:
        matches = self.search(text_vector, k=k, require_image=True, exclude_id=exclude_id)
        if not matches:
            return np.zeros(self.dim, dtype=np.float32)
        return _unit(np.mean([hit["data_vector"] for hit in matches], axis=0))
