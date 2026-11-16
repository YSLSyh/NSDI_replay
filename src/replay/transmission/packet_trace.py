"""将有逐包丢失观测的 5G 轨迹转换成可重复的应用层丢包事件。

本模块只接受每个**已尝试发送**的包都有一行、且显式记录丢失与否的轨迹。
只有吞吐量或接收端抓包的文件不能证明哪些包丢失，必须先用发送端序号
与接收端记录配对，生成规范 CSV。它不推断缺失事件，也不合成 5G 测量值。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path


_LOSS_EVIDENCE = frozenset({"explicit_packet_outcome", "paired_sender_receiver_logs"})


@dataclass(frozen=True)
class TraceProvenance:
    """数据来源及逐包丢失的证据；填写内容应可由原始数据核查。"""

    dataset: str
    source_uri: str
    network: str
    capture_method: str
    loss_evidence: str
    license: str

    def __post_init__(self) -> None:
        for key in ("dataset", "source_uri", "network", "capture_method", "license"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"轨迹来源缺少 {key}")
        if not isinstance(self.loss_evidence, str) or self.loss_evidence not in _LOSS_EVIDENCE:
            raise ValueError("loss_evidence 必须说明显式丢包事件或发送/接收日志配对")


@dataclass(frozen=True)
class PacketTraceEvent:
    """一行对应一次尝试发送，而不只是一次成功接收。"""

    timestamp_ms: float
    lost: bool
    throughput_mbps: float | None


@dataclass(frozen=True)
class PacketTrace:
    path: Path
    sha256: str
    provenance: TraceProvenance
    events: tuple[PacketTraceEvent, ...]

    @property
    def loss_rate(self) -> float:
        return sum(event.lost for event in self.events) / len(self.events)

    def select_segment(
        self, *, n_packets: int, window_packets: int, seed: int,
        case_id: str, start_index: int | None = None,
    ) -> TraceReplay:
        """选择同一案例共享的轨迹窗口，再取前 ``n_packets`` 个事件。

        比较不同 codec 时，先算出各方法最大包数，统一传入 ``window_packets``。
        这样各方法虽有不同码长，仍从同一个时间点、同一丢包序列前缀开始。
        不循环、扩展或补造轨迹；轨迹不够长时直接报错。
        """
        for name, value in (("n_packets", n_packets), ("window_packets", window_packets)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if n_packets > window_packets or window_packets > len(self.events):
            raise ValueError("请求的轨迹窗口超过可用的逐包事件")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed 必须是整数")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id 必须是非空字符串")
        choices = len(self.events) - window_packets + 1
        if start_index is None:
            key = json.dumps([seed, case_id, self.sha256],
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            start_index = int.from_bytes(sha256(key).digest()[:8], "big") % choices
        elif (not isinstance(start_index, int) or isinstance(start_index, bool)
              or start_index < 0 or start_index >= choices):
            raise ValueError("start_index 不在完整轨迹窗口的范围内")
        sample = self.events[start_index:start_index + n_packets]
        return TraceReplay(
            trace_sha256=self.sha256,
            provenance=self.provenance,
            start_index=start_index,
            window_packets=window_packets,
            loss_trace=tuple(event.lost for event in sample),
            timestamps_ms=tuple(event.timestamp_ms for event in sample),
            observed_throughput_mbps=tuple(event.throughput_mbps for event in sample),
        )


@dataclass(frozen=True)
class TraceReplay:
    """可直接作为现有 ``loss_trace`` 参数输入的实测丢包事件。"""

    trace_sha256: str
    provenance: TraceProvenance
    start_index: int
    window_packets: int
    loss_trace: tuple[bool, ...]
    timestamps_ms: tuple[float, ...]
    observed_throughput_mbps: tuple[float | None, ...]

    @property
    def duration_ms(self) -> float:
        return self.timestamps_ms[-1] - self.timestamps_ms[0]

    @property
    def loss_rate(self) -> float:
        return sum(self.loss_trace) / len(self.loss_trace)

    def metadata(self) -> dict[str, object]:
        return {
            "trace_sha256": self.trace_sha256,
            "dataset": self.provenance.dataset,
            "source_uri": self.provenance.source_uri,
            "network": self.provenance.network,
            "capture_method": self.provenance.capture_method,
            "loss_evidence": self.provenance.loss_evidence,
            "license": self.provenance.license,
            "start_index": self.start_index,
            "window_packets": self.window_packets,
            "n_packets": len(self.loss_trace),
            "lost_packets": sum(self.loss_trace),
            "actual_loss_rate": self.loss_rate,
            "observed_duration_ms": self.duration_ms,
            "has_observed_throughput": all(value is not None
                                           for value in self.observed_throughput_mbps),
        }


def _number(value: str | None, *, name: str, row: int,
            positive: bool = False) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"第 {row} 行 {name} 不是数字") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(f"第 {row} 行 {name} 必须为有限{'正数' if positive else '非负数'}")
    return number


def _lost(value: str | None, *, row: int) -> bool:
    if value is None:
        raise ValueError(f"第 {row} 行缺少显式丢包事件")
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"第 {row} 行 lost 必须为 0/1 或 true/false")


def load_packet_trace_csv(path: str | Path, *, provenance: TraceProvenance) -> PacketTrace:
    """读取规范 CSV：``timestamp_ms,lost[,throughput_mbps]``。

    timestamp 是发送尝试的时刻；每行均须有显式 lost 标志。可选吞吐量为空时
    保留 ``None``，不会用随机值填充。原始文件 SHA256 写入每次回放元数据。
    """
    source = Path(path).resolve()
    raw = source.read_bytes()
    digest = sha256(raw).hexdigest()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("轨迹 CSV 必须使用 UTF-8 编码") from exc
    reader = csv.DictReader(content.splitlines())
    names = reader.fieldnames
    if names is None or len(names) != len(set(names)):
        raise ValueError("轨迹 CSV 缺少表头或有重复列")
    if not {"timestamp_ms", "lost"}.issubset(names):
        raise ValueError("需要逐包 timestamp_ms 与显式 lost；仅吞吐量轨迹不能推断丢包")
    events: list[PacketTraceEvent] = []
    previous_timestamp = -1.0
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"第 {row_number} 行列数与表头不符")
        if "packet_index" in names:
            try:
                packet_index = int(row["packet_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {row_number} 行 packet_index 不是整数") from exc
            if packet_index != row_number - 2:
                raise ValueError("packet_index 必须从 0 连续递增，且每次发送都有一行")
        timestamp = _number(row["timestamp_ms"], name="timestamp_ms", row=row_number)
        if timestamp < previous_timestamp:
            raise ValueError(f"第 {row_number} 行时间戳倒退")
        previous_timestamp = timestamp
        throughput_raw = row.get("throughput_mbps")
        throughput = (_number(throughput_raw, name="throughput_mbps",
                              row=row_number, positive=True)
                      if throughput_raw is not None and throughput_raw.strip() else None)
        events.append(PacketTraceEvent(timestamp, _lost(row["lost"], row=row_number),
                                       throughput))
    if not events:
        raise ValueError("轨迹 CSV 没有逐包事件")
    return PacketTrace(source, digest, provenance, tuple(events))


def load_trace_provenance_json(path: str | Path) -> TraceProvenance:
    """从伴随 CSV 的 JSON 读取可核查来源，不把无来源文件标记为实测轨迹。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"dataset", "source_uri", "network", "capture_method",
              "loss_evidence", "license"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError(f"来源 JSON 必须且只能包含 {sorted(fields)}")
    return TraceProvenance(**data)
