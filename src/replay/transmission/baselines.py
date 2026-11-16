"""图像基线：标准 JPEG 与 Progressive JPEG，均保持原始图像尺寸。"""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def encode_jpeg_baseline(image: Image.Image, quality: int, subsampling: int,
                         *, progressive: bool = False,
                         restart_marker_rows: int = 0,
                         restart_marker_blocks: int = 0,
                         optimize_huffman: bool = False,
                         allow_420: bool = False) -> bytes:
    """用同一个 libjpeg 后端生成普通或渐进式 JPEG。"""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality 必须在 1–100 之间")
    if subsampling not in (-1, 0, 1, 2):
        raise ValueError("JPEG subsampling 必须为 -1、0、1 或 2")
    if restart_marker_rows not in (0, 1):
        raise ValueError("当前传输实验仅支持 restart_marker_rows=1")
    if restart_marker_blocks < 0 or (restart_marker_rows and restart_marker_blocks):
        raise ValueError("JPEG restart MCU 间隔无效")
    if (restart_marker_rows or restart_marker_blocks) and not (
            subsampling == 0 or subsampling == 2 and allow_420):
        raise ValueError("按 MCU 区间分包要求 JPEG 4:4:4 (subsampling=0)")
    if not isinstance(optimize_huffman, bool):
        raise TypeError("optimize_huffman 必须是布尔值")
    stream = BytesIO()
    options = {"format": "JPEG", "quality": quality, "subsampling": subsampling,
               "progressive": progressive, "optimize": optimize_huffman}
    if restart_marker_rows:
        options["restart_marker_rows"] = restart_marker_rows
    if restart_marker_blocks:
        options["restart_marker_blocks"] = restart_marker_blocks
    image.convert("RGB").save(stream, **options)
    return stream.getvalue()


def encode_jpeg_baseline_under_budget(
        image: Image.Image, budget_bytes: int, subsampling: int,
        *, progressive: bool = False, max_quality: int = 95,
        restart_marker_rows: int = 0,
        packet_header_bytes: int = 0,
        restart_marker_blocks: int = 0,
        packet_manifest_bytes_per_interval: int = 0,
        optimize_huffman: bool = False,
        allow_420: bool = False) -> tuple[bytes, dict]:
    """按实际码流和可选的每包传输头搜索质量；不调整图像尺寸。"""
    if budget_bytes <= 0:
        raise ValueError("JPEG 字节预算必须为正")
    if not 1 <= max_quality <= 100:
        raise ValueError("max_quality 必须在 1–100 之间")
    if packet_header_bytes < 0 or packet_manifest_bytes_per_interval < 0:
        raise ValueError("传输开销不能为负")
    if (packet_header_bytes or packet_manifest_bytes_per_interval) and not (
            restart_marker_rows or restart_marker_blocks):
        raise ValueError("按包计预算时必须设置 JPEG restart marker")
    cache: dict[int, bytes] = {}

    def wire_size(payload: bytes) -> int:
        if not packet_header_bytes and not packet_manifest_bytes_per_interval:
            return len(payload)
        from replay.transmission.packet_loss import (count_entropy_intervals,
                                                     count_entropy_packets)
        return (len(payload) + packet_header_bytes * count_entropy_packets(
            payload, allow_420=allow_420)
                + packet_manifest_bytes_per_interval * count_entropy_intervals(
                    payload, allow_420=allow_420))

    def encoded(quality: int) -> bytes:
        if quality not in cache:
            cache[quality] = encode_jpeg_baseline(
                image, quality, subsampling, progressive=progressive,
                restart_marker_rows=restart_marker_rows,
                restart_marker_blocks=restart_marker_blocks,
                optimize_huffman=optimize_huffman,
                allow_420=allow_420)
        return cache[quality]

    # 明确检查质量 1：若原尺寸仍超预算，直接报告不可行，不偷偷缩图。
    minimum = encoded(1)
    if wire_size(minimum) > budget_bytes:
        raise ValueError(
            f"原尺寸下无法满足 {budget_bytes} 字节预算；"
            f"quality=1 需 {wire_size(minimum)} 字节")
    best_quality = 1
    low, high = 2, max_quality
    while low <= high:
        quality = (low + high) // 2
        if wire_size(encoded(quality)) <= budget_bytes:
            best_quality = quality
            low = quality + 1
        else:
            high = quality - 1
    # 文件长度偶尔不严格单调；检查相邻质量，仍以真实字节数为准。
    for quality in range(best_quality + 1, min(max_quality, best_quality + 3) + 1):
        if wire_size(encoded(quality)) <= budget_bytes:
            best_quality = quality
    payload = encoded(best_quality)
    return payload, {"jpeg_bytes": len(payload), "jpeg_quality": best_quality,
                     "codec_wire_bytes": wire_size(payload),
                     "packet_header_bytes": packet_header_bytes,
                     "packet_manifest_bytes_per_interval": packet_manifest_bytes_per_interval,
                     "restart_marker_rows": restart_marker_rows,
                     "restart_marker_blocks": restart_marker_blocks,
                     "optimize_huffman": optimize_huffman,
                     "allow_420": allow_420,
                     "quality": best_quality,
                     "progressive": progressive, "budget_bytes": budget_bytes,
                     "original_size": [image.width, image.height]}


def encode_baseline_fixed_quality(
        image: Image.Image, method: str, quality: int = 75,
        subsampling: int = 0, *, optimize_huffman: bool = False) -> bytes:
    """按指定质量编码标准 JPEG 或 Progressive JPEG。"""
    if method not in ("jpeg", "progressive_jpeg"):
        raise ValueError(f"不支持的图像基线: {method}")
    return encode_jpeg_baseline(
        image, quality, subsampling, progressive=method == "progressive_jpeg",
        optimize_huffman=optimize_huffman)


def encode_baseline_under_budget(
        image: Image.Image, method: str, budget_bytes: int,
        subsampling: int = 0, max_quality: int = 95,
        restart_marker_rows: int = 0,
        packet_header_bytes: int = 0,
        restart_marker_blocks: int = 0,
        packet_manifest_bytes_per_interval: int = 0,
        optimize_huffman: bool = False,
        allow_420: bool = False) -> tuple[bytes, dict]:
    """对真实 JPEG 码流做字节预算搜索；预算由调用方扣除传输开销。"""
    if method not in ("jpeg", "progressive_jpeg"):
        raise ValueError(f"不支持的图像基线: {method}")
    payload, info = encode_jpeg_baseline_under_budget(
        image, budget_bytes, subsampling,
        progressive=method == "progressive_jpeg", max_quality=max_quality,
        restart_marker_rows=restart_marker_rows,
        packet_header_bytes=packet_header_bytes,
        restart_marker_blocks=restart_marker_blocks,
        packet_manifest_bytes_per_interval=packet_manifest_bytes_per_interval,
        optimize_huffman=optimize_huffman, allow_420=allow_420)
    return payload, {**info, "method": method}
