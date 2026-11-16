"""核验 Cityscapes train 的多维 JPEG 参数消融报告，供正式 val 冻结发送端。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from replay.core.config import ReplayConfig
from replay.evaluation.figure8a_dct_ablation import (
    DEFAULT_CANDIDATES, TARGET_RAW_IDS, summarize_dct_rows,
)


def load_dct_sender_candidate(config: ReplayConfig, report_path: Path, *,
                              task: str, budget_bytes: int,
                              train_image_ids: set[str]) -> tuple[dict, str]:
    """至少 50 张/类、完整候选记录和原尺寸 30 kB，任何 val 数据都不参与选择。"""
    if task not in TARGET_RAW_IDS:
        raise ValueError("未知目标类别")
    raw = report_path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    settings = report.get("settings", {})
    expected = {
        "protocol": "figure8a_train_dct_ablation_v1", "split": "train",
        "pilot": False, "budget_wire_bytes": 30_000,
        "native_resolution": [2048, 1024],
        "jpeg_subsampling": "candidate-dependent 4:4:4 or 4:2:0",
        "optimize_huffman": True,
        "restart_blocks": "candidate-dependent 2048 or 512 MCU",
        "packet_payload_bytes": 1200, "packet_header_bytes": 8,
        "no_loss_codec_only": True,
        "segmenter_revision": config.section("models")["segmenter"]["revision"],
        "config_sha256": sha256(json.dumps(config.values, sort_keys=True,
                                            default=str).encode()).hexdigest(),
    }
    if budget_bytes != 30_000 or any(settings.get(key) != value
                                     for key, value in expected.items()):
        raise ValueError("DCT 训练报告的划分、预算、分辨率、编码器或模型配置不匹配")
    count = settings.get("sample_images_per_task")
    if type(count) is not int or count < 50:
        raise ValueError("正式评测要求 DCT 训练报告至少每类 50 张正样本")
    canonical = {candidate["id"]: candidate for candidate in DEFAULT_CANDIDATES}
    candidates = settings.get("candidates")
    if (not isinstance(candidates, list) or not candidates or
            any(not isinstance(item, dict) or
                item != canonical.get(item.get("id")) for item in candidates) or
            len({item["id"] for item in candidates}) != len(candidates)):
        raise ValueError("DCT 训练报告的候选配置不在预注册列表内或存在重复")
    sampled = settings.get("sampled_image_ids")
    if not isinstance(sampled, dict) or set(sampled) != set(TARGET_RAW_IDS):
        raise ValueError("DCT 训练报告缺少三个类别的样本清单")
    for name, ids in sampled.items():
        if (not isinstance(ids, list) or len(ids) != count or
                len(set(ids)) != count or not set(ids).issubset(train_image_ids)):
            raise ValueError(f"{name} 的训练集图像 ID 无法核实")
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("DCT 训练报告缺少逐图记录")
    seen: set[tuple[str, str, str]] = set()
    candidate_ids = {item["id"] for item in candidates}
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("DCT 训练报告逐图记录格式错误")
        name = row.get("task")
        candidate_id = row.get("candidate_id")
        image_id = row.get("image_id")
        key = name, candidate_id, image_id
        counts = row.get("counts")
        status = row.get("status")
        if (name not in TARGET_RAW_IDS or candidate_id not in candidate_ids or
                image_id not in sampled[name] or key in seen or
                status not in {"ok", "infeasible"} or
                not isinstance(counts, list) or len(counts) != 3 or
                any(type(value) is not int or value < 0 for value in counts)):
            raise ValueError("DCT 训练报告逐图记录重复、缺失或数值无效")
        if status == "infeasible" and (counts[0] != 0 or counts[1] != 0):
            raise ValueError("无法编码的图像未按目标漏检计分")
        if status == "ok" and (
                type(row.get("sent_bytes")) is not int or
                row["sent_bytes"] > budget_bytes or
                type(row.get("jpeg_quality")) is not int or
                not 1 <= row["jpeg_quality"] <= 95):
            raise ValueError("DCT 训练报告成功记录的字节或质量无效")
        seen.add(key)
    expected_keys = {(name, item["id"], image_id)
                     for name, ids in sampled.items()
                     for item in candidates for image_id in ids}
    if seen != expected_keys:
        raise ValueError("DCT 训练报告未覆盖三类全部训练图像和候选")
    recalculated = summarize_dct_rows(runs, tuple(candidates), count)
    provided = report.get("summary")
    if not isinstance(provided, dict) or set(provided) != set(TARGET_RAW_IDS):
        raise ValueError("DCT 训练报告的选择结果缺失")
    for name in TARGET_RAW_IDS:
        if provided[name].get("selected_candidate_id") != recalculated[name]["selected_candidate_id"]:
            raise ValueError("DCT 训练报告的候选选择与逐图计数不一致")
    chosen_id = recalculated[task]["selected_candidate_id"]
    return canonical[chosen_id].copy(), sha256(raw).hexdigest()
