"""在评测时保存真正参与 IoU 计数的少量案例，供汇报图无损重现。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from replay.evaluation.figure8 import binary_counts


LEGACY_CAPTURE_PROTOCOL = "figure8a_exact_eval_capture_v1"
CAPTURE_PROTOCOL = "figure8a_exact_eval_capture_v2"
BASE_ROLES = ("best_vs_limitnet", "worst_vs_limitnet", "best_vs_jpeg")
ROLES = BASE_ROLES + ("best_lossy_kdc_vs_jpeg", "best_restoration_gain")
METHODS = ("jpeg", "limitnet", "kdc_direct", "kdc_restored")


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iou(row: dict) -> float:
    if row["status"] != "ok":
        return 0.0
    tp, fp, fn = row["counts"]
    return float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0


def capture_role_scores(rows: dict[str, dict], limit_method: str, *,
                        include_lossy: bool = True) -> dict[str, float]:
    """失败帧按主评测同样记零；JPEG 对比仅选择双方都可解码的图。"""
    if rows["kdc_restored"]["gt_target_pixels"] <= 0:
        return {}
    restored = _iou(rows["kdc_restored"])
    delta_limit = restored - _iou(rows[limit_method])
    scores = {"best_vs_limitnet": delta_limit,
              "worst_vs_limitnet": delta_limit}
    if rows["kdc_restored"]["status"] == rows["jpeg"]["status"] == "ok":
        delta_jpeg = restored - _iou(rows["jpeg"])
        if delta_jpeg > 0:
            scores["best_vs_jpeg"] = delta_jpeg
    # 新增案例必须来自观测轨迹里的实际丢包，不能拿零丢包图解释恢复效果。
    if include_lossy and int(rows["kdc_direct"].get("lost_packets") or 0) > 0:
        if rows["kdc_restored"]["status"] == rows["jpeg"]["status"] == "ok":
            scores["best_lossy_kdc_vs_jpeg"] = restored - _iou(rows["jpeg"])
        if (rows["kdc_direct"]["status"] == rows["kdc_restored"]["status"] == "ok"
                and bool(rows["kdc_restored"].get("receiver_applied"))):
            scores["best_restoration_gain"] = restored - _iou(rows["kdc_direct"])
    return scores


def _better(role: str, score: float, image_id: str, previous: dict | None) -> bool:
    if previous is None:
        return True
    old_score = float(previous["score"])
    old_id = str(previous["image_id"])
    if role == "worst_vs_limitnet":
        return score < old_score or (score == old_score and image_id > old_id)
    return (-score, image_id) < (-old_score, old_id)


class ExactCaseCapture:
    """仅维护三个预定义候选案例；原图和预测 Mask 来自同一次分割调用。"""

    def __init__(self, output: Path, settings: dict, *, existing_rows: int) -> None:
        self.output = Path(output)
        self.directory = self.output / "captured_cases"
        self.path = self.directory / "manifest.json"
        self.limit_method = next(name for name in settings["methods"]
                                 if name.startswith("limitnet_"))
        identity = {key: settings[key] for key in (
            "task", "class_id", "trace_sha256", "config_sha256",
            "budget_wire_bytes", "limitnet_checkpoint_sha256",
            "receiver_checkpoint_sha256", "seed")}
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
            if self.manifest.get("protocol") != CAPTURE_PROTOCOL or self.manifest.get("identity") != identity:
                raise ValueError("既有案例捕获与当前评测配置不一致")
        elif existing_rows:
            raise ValueError("已有逐图记录但缺少同期捕获文件；请使用新的输出目录评测")
        else:
            self.manifest = {"protocol": CAPTURE_PROTOCOL, "identity": identity,
                             "selection": "全验证集在线选 best/worst KDC-LimitNet、正提升 KDC-JPEG、实测 KDC 丢包且双方可解码的 best KDC-JPEG、以及实际调用恢复网络的最大恢复增益；后两项若得分不正仍如实保留",
                             "roles": {}}
            self.directory.mkdir(parents=True, exist_ok=True)

    def _save_image(self, folder: Path, name: str, image: Image.Image) -> dict:
        path = folder / f"{name}.png"
        temporary = folder / f"{name}.tmp"
        image.save(temporary, format="PNG", compress_level=3)
        temporary.replace(path)
        return {"path": str(path.relative_to(self.output)),
                "sha256": _file_digest(path)}

    def consider(self, *, image_id: str, original: Image.Image,
                 truth: np.ndarray, missing: np.ndarray,
                 pictures: dict[str, Image.Image | None],
                 predictions: dict[str, np.ndarray | None],
                 rows: dict[str, dict]) -> None:
        if any(name not in rows for name in ("jpeg", self.limit_method,
                                              "kdc_direct", "kdc_restored")):
            raise ValueError("案例捕获要求该图四个方法均已完成评测")
        scores = capture_role_scores(rows, self.limit_method)
        for role, score in scores.items():
            previous = self.manifest["roles"].get(role)
            if not _better(role, score, image_id, previous):
                continue
            folder = self.directory / role
            folder.mkdir(parents=True, exist_ok=True)
            files: dict[str, dict] = {
                "original": self._save_image(folder, "original", original.convert("RGB")),
                "truth_mask": self._save_image(
                    folder, "truth_mask", Image.fromarray(np.asarray(truth, dtype=np.uint8), "L")),
                "kdc_missing": self._save_image(
                    folder, "kdc_missing",
                    Image.fromarray((np.asarray(missing) >= 0.5).astype(np.uint8) * 255, "L")),
            }
            for method in METHODS:
                row_name = self.limit_method if method == "limitnet" else method
                candidate = pictures.get(method)
                prediction = predictions.get(method)
                if rows[row_name]["status"] == "ok":
                    if candidate is None or prediction is None:
                        raise ValueError(f"{image_id}/{method} 的已评分图像或 Mask 缺失")
                    if list(binary_counts(prediction, truth,
                                          int(self.manifest["identity"]["class_id"]))) != rows[row_name]["counts"]:
                        raise ValueError(f"{image_id}/{method} 捕获的预测 Mask 与评测计数不一致")
                    files[method] = self._save_image(folder, method, candidate.convert("RGB"))
                    files[f"{method}_mask"] = self._save_image(
                        folder, f"{method}_mask",
                        Image.fromarray(np.asarray(prediction, dtype=np.uint8), "L"))
                elif candidate is not None or prediction is not None:
                    raise ValueError(f"{image_id}/{method} 失败却提供了已评分图像")
            self.manifest["roles"][role] = {
                "image_id": image_id, "score": float(score), "files": files,
                "rows": {method: {"status": rows[self.limit_method if method == "limitnet" else method]["status"],
                                  "counts": rows[self.limit_method if method == "limitnet" else method]["counts"],
                                  "trace_start_index": rows[self.limit_method if method == "limitnet" else method].get("trace_start_index"),
                                  "lost_packets": rows[self.limit_method if method == "limitnet" else method].get("lost_packets"),
                                  "receiver_applied": rows[self.limit_method if method == "limitnet" else method].get("receiver_applied")}
                         for method in METHODS},
            }
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.manifest, ensure_ascii=False,
                                             indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)

