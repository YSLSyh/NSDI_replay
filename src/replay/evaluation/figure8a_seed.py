"""为独立接收端消融预填不依赖接收端的 Fig8(a) 逐图结果。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from replay.evaluation.figure8a_packet import (TARGETS,
                                               methods_for_limitnet_mode)


SEED_PROTOCOL = "figure8a_adapted_receiver_seed_v1"
SOURCE_PROTOCOL = "figure8a_cityscapes_packet_v10_codec_selected"
SHARED_SETTINGS_KEYS = (
    "task", "class_id", "split", "native_resolution", "image_ids",
    "full_validation", "pilot", "seed", "budget_wire_bytes", "methods",
    "limitnet_mode", "trace_sha256", "trace_provenance", "trace_mapping",
    "trace_reserved_window_packets", "source_packet_bytes",
    "packet_size_mismatch", "allow_packet_size_mismatch",
    "packet_payload_bytes", "packet_header_bytes", "jpeg_restart_blocks",
    "jpeg_optimize_huffman", "kdc_background_lowpass_sigma",
    "sender_sigma_source", "sender_tuning_report_sha256",
    "kdc_codec_candidate", "config_sha256", "kb_path",
    "limitnet_checkpoint_sha256", "segmenter_revision", "segmenter_tile",
    "segmenter_stride",
)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_rows(report: dict, runs_bytes: bytes, *, task: str,
                 seed: int) -> tuple[list[tuple[dict, bytes]], dict]:
    """完整核对原报告与原 JSONL，保留每行原始字节。"""
    settings = report.get("settings", {})
    summary = report.get("summary", {})
    if (settings.get("protocol") != SOURCE_PROTOCOL or
            settings.get("task") != task or settings.get("class_id") != TARGETS[task] or
            type(settings.get("seed")) is not int or settings["seed"] != seed or
            settings.get("split") != "val" or
            settings.get("full_validation") is not True or
            settings.get("pilot") is not False or
            settings.get("budget_wire_bytes") != 30_000 or
            not isinstance(settings.get("kdc_codec_candidate"), dict) or
            settings["kdc_codec_candidate"].get("subsampling") != 2):
        raise ValueError("来源必须是同任务、同 seed 的正式 30 kB、4:2:0 主接收端全验证报告")
    image_ids = settings.get("image_ids")
    if (not isinstance(image_ids, list) or len(image_ids) != 500 or
            len(set(image_ids)) != 500 or
            not all(isinstance(value, str) for value in image_ids)):
        raise ValueError("来源必须包含 500 个唯一 Cityscapes val 图像 ID")
    expected_methods = methods_for_limitnet_mode(settings.get("limitnet_mode"))
    if settings.get("methods") != list(expected_methods):
        raise ValueError("来源报告的方法集合与 LimitNet 模式不一致")
    if (summary.get("task") != task or summary.get("class_id") != TARGETS[task] or
            summary.get("n_expected_images") != 500 or
            summary.get("n_complete_images") != 500 or
            summary.get("all_expected_complete") is not True):
        raise ValueError("来源报告尚未完整评测 500 张图像")
    if any(key not in settings for key in SHARED_SETTINGS_KEYS):
        raise ValueError("来源报告缺少可核对的实验设置")
    report_rows = report.get("runs")
    if not isinstance(report_rows, list) or len(report_rows) != 500 * len(expected_methods):
        raise ValueError("来源报告未包含每图四方法的完整逐图记录")
    lines = runs_bytes.splitlines(keepends=True)
    if len(lines) != len(report_rows) or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("来源 runs.jsonl 行数或行尾与报告不一致")
    rows: list[tuple[dict, bytes]] = []
    keys: set[tuple[str, str]] = set()
    expected_keys = {(image_id, method) for image_id in image_ids
                     for method in expected_methods}
    for line, reported in zip(lines, report_rows):
        row = json.loads(line.decode("utf-8"))
        key = (row.get("image_id"), row.get("method"))
        if (key not in expected_keys or key in keys or row != reported or
                row.get("task") != task or row.get("class_id") != TARGETS[task] or
                row.get("budget_wire_bytes") != 30_000 or
                row.get("trace_sha256") != settings["trace_sha256"]):
            raise ValueError("来源 runs.jsonl 与完整报告或共同实验身份不一致")
        keys.add(key)
        rows.append((row, line))
    if keys != expected_keys:
        raise ValueError("来源逐图记录没有覆盖全部 500×4 个键")
    by_key = {(row["image_id"], row["method"]): row for row, _ in rows}
    for image_id in image_ids:
        direct = by_key[(image_id, "kdc_direct")]
        restored = by_key[(image_id, "kdc_restored")]
        if restored["status"] == "ok" and restored.get("receiver_applied") is False:
            if (direct["status"] != "ok" or
                    restored["counts"] != direct["counts"] or
                    ("restored_image_sha256" in restored and
                     "decoded_image_sha256" in direct and
                     restored["restored_image_sha256"] !=
                     direct["decoded_image_sha256"])):
                raise ValueError("未调用接收端的恢复行与直接解码结果不一致，不能复用")
        elif restored["status"] == "ok" and restored.get("receiver_applied") is not True:
            raise ValueError("可解码恢复行未明确记录 receiver_applied")
    return rows, settings


def seed_adapted_receiver_eval(*, source_report: Path, source_runs: Path,
                               receiver_checkpoint: Path, output: Path,
                               task: str, seed: int,
                               receiver_mask_only: bool = False) -> dict:
    """复制原始 JSONL 字节；仅省略实际调用旧接收端的恢复行。"""
    if task not in TARGETS:
        raise ValueError("未知类别")
    if output.exists():
        raise ValueError("预填目标必须是尚不存在的新目录，防止混入既有评测")
    report_bytes = source_report.read_bytes()
    report = json.loads(report_bytes)
    runs_bytes = source_runs.read_bytes()
    rows, settings = _source_rows(report, runs_bytes, task=task, seed=seed)
    kept: list[bytes] = []
    omitted: list[str] = []
    for row, line in rows:
        if row["method"] == "kdc_restored" and row.get("receiver_applied") is True:
            if (row["status"] != "ok" or
                    int(row.get("lost_packets") or 0) <= 0 or
                    int(row.get("lost_intervals") or 0) <= 0):
                raise ValueError("receiver_applied=True 的来源恢复行必须有实际丢包且可解码")
            omitted.append(row["image_id"])
        else:
            kept.append(line)
    if not omitted:
        raise ValueError("来源没有实际调用接收端的恢复行，无法构成适配消融")
    seeded_bytes = b"".join(kept)
    receiver_digest = _digest(receiver_checkpoint)
    if receiver_digest == settings["receiver_checkpoint_sha256"]:
        raise ValueError("适配接收端必须与主报告的原始接收端权重不同")
    provenance = {
        "protocol": SEED_PROTOCOL,
        "source_report": str(source_report.resolve()),
        "source_report_sha256": sha256(report_bytes).hexdigest(),
        "source_runs": str(source_runs.resolve()),
        "source_runs_sha256": sha256(runs_bytes).hexdigest(),
        "source_receiver_checkpoint_sha256": settings["receiver_checkpoint_sha256"],
        "adapted_receiver_checkpoint_sha256": receiver_digest,
        "receiver_output_mode": ("mask_only" if receiver_mask_only else "full_rgb"),
        "source_settings_identity": {key: settings[key] for key in SHARED_SETTINGS_KEYS},
        "source_rows": len(rows), "seeded_rows": len(kept),
        "omitted_receiver_applied_restored_rows": len(omitted),
        "omitted_image_ids": omitted,
        "seeded_prefix_bytes": len(seeded_bytes),
        "seeded_prefix_sha256": sha256(seeded_bytes).hexdigest(),
        "case_capture": "disabled; exact-case capture from the main receiver cannot be reused",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs.jsonl").write_bytes(seeded_bytes)
    (output / "seed_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return provenance


def validate_seed_for_settings(output: Path, settings: dict) -> dict:
    """续跑前校验原始行字节前缀与所有共享配置，防止错配来源。"""
    provenance = json.loads((output / "seed_provenance.json").read_text(encoding="utf-8"))
    mode = provenance.get("receiver_output_mode")
    expected_protocol = "figure8a_cityscapes_packet_v10_codec_selected_receiver_adapted"
    if mode == "mask_only":
        expected_protocol += "_mask_only"
        expected_policy = "replace_missing_entropy_intervals_only_no_halo"
    elif mode == "full_rgb":
        expected_policy = None
    else:
        raise ValueError("预填记录未标明有效的接收端输出策略")
    if (provenance.get("protocol") != SEED_PROTOCOL or
            settings.get("case_capture") != "disabled_explicit_adapted_ablation" or
            settings.get("protocol") != expected_protocol or
            settings.get("receiver_output_policy") != expected_policy):
        raise ValueError("预填记录与关闭案例捕获的适配接收端输出策略不一致")
    if any(settings.get(key) != value for key, value in
           provenance["source_settings_identity"].items()):
        raise ValueError("预填逐图记录与当前评测的任务、种子、图像、轨迹或模型设置不一致")
    if (settings.get("receiver_checkpoint_sha256") !=
            provenance.get("adapted_receiver_checkpoint_sha256") or
            settings.get("receiver_adaptation_metadata", {}).get(
                "initial_receiver_sha256") !=
            provenance.get("source_receiver_checkpoint_sha256")):
        raise ValueError("预填记录的主接收端/适配接收端权重与当前评测不一致")
    length = provenance["seeded_prefix_bytes"]
    if type(length) is not int or length <= 0:
        raise ValueError("预填记录的字节长度无效")
    with (output / "runs.jsonl").open("rb") as source:
        prefix = source.read(length)
    if (len(prefix) != length or sha256(prefix).hexdigest() !=
            provenance["seeded_prefix_sha256"]):
        raise ValueError("预填逐图记录原始字节前缀已改变")
    return provenance
