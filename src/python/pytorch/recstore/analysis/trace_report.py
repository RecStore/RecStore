from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

TRACE_CSV_FIELDS = [
    "trace_path",
    "rank",
    "step",
    "collective_total_ms",
    "nccl_kernel_ms",
    "cuda_stream_sync_ms",
    "unknown_sync_ms",
    "input_dist_pack_ms",
    "output_dist_unpack_ms",
    "unclassified_collective_ms",
]


@dataclass
class TraceSummary:
    trace_path: str
    rank: int
    step: int
    collective_total_ms: float
    nccl_kernel_ms: float
    cuda_stream_sync_ms: float
    unknown_sync_ms: float
    input_dist_pack_ms: float
    output_dist_unpack_ms: float
    unclassified_collective_ms: float

    def as_row(self) -> dict[str, float | str]:
        return {
            "trace_path": self.trace_path,
            "rank": self.rank,
            "step": self.step,
            "collective_total_ms": self.collective_total_ms,
            "nccl_kernel_ms": self.nccl_kernel_ms,
            "cuda_stream_sync_ms": self.cuda_stream_sync_ms,
            "unknown_sync_ms": self.unknown_sync_ms,
            "input_dist_pack_ms": self.input_dist_pack_ms,
            "output_dist_unpack_ms": self.output_dist_unpack_ms,
            "unclassified_collective_ms": self.unclassified_collective_ms,
        }


def _extract_events(trace: object) -> list[dict]:
    if isinstance(trace, dict):
        events = trace.get("traceEvents", [])
        if isinstance(events, list):
            return [ev for ev in events if isinstance(ev, dict)]
        return []
    if isinstance(trace, list):
        return [ev for ev in trace if isinstance(ev, dict)]
    return []


def _event_name(event: dict) -> str:
    name = event.get("name", "")
    if not isinstance(name, str):
        return ""
    return name


def _event_duration_ms(event: dict) -> float:
    dur = event.get("dur")
    if isinstance(dur, (int, float)):
        return float(dur) / 1000.0
    return 0.0


def _matches_any(lower_name: str, needles: list[str]) -> bool:
    return any(needle in lower_name for needle in needles)


def _extract_rank_step(path: Path) -> tuple[int, int]:
    stem = path.name.lower()
    rank = 0
    step = -1

    rank_match = re.search(r"(?:^|[_\-.])rank[_\-.]?(\d+)(?:[_\-.]|$)", stem)
    if rank_match is not None:
        rank = int(rank_match.group(1))

    step_match = re.search(r"(?:^|[_\-.])step[_\-.]?(\d+)(?:[_\-.]|$)", stem)
    if step_match is not None:
        step = int(step_match.group(1))

    return rank, step


def summarize_trace_file(path: Path) -> dict[str, float | str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = _extract_events(data)
    rank, step = _extract_rank_step(path)
    nccl_kernel_ms = 0.0
    cuda_stream_sync_ms = 0.0
    unknown_sync_ms = 0.0
    input_dist_pack_ms = 0.0
    output_dist_unpack_ms = 0.0
    unclassified_collective_ms = 0.0

    for event in events:
        name = _event_name(event)
        dur_ms = _event_duration_ms(event)
        if dur_ms <= 0.0:
            continue
        lower_name = name.lower()

        if _matches_any(lower_name, ["input_dist"]):
            input_dist_pack_ms += dur_ms
            continue
        if _matches_any(lower_name, ["output_dist"]):
            output_dist_unpack_ms += dur_ms
            continue
        if _matches_any(lower_name, ["cudastreamsynchronize", "cudadevicesynchronize"]):
            cuda_stream_sync_ms += dur_ms
            continue
        if _matches_any(lower_name, ["synchronize", "wait"]):
            unknown_sync_ms += dur_ms
            continue
        if _matches_any(lower_name, ["nccl", "all_to_all", "alltoall"]):
            nccl_kernel_ms += dur_ms
            continue
        if _matches_any(
            lower_name,
            [
                "collective",
                "distributed_c10d",
                "all_to_all",
                "all_reduce",
                "reduce_scatter",
                "all_gather",
                "broadcast",
            ],
        ):
            unclassified_collective_ms += dur_ms

    collective_total_ms = nccl_kernel_ms + unclassified_collective_ms
    summary = TraceSummary(
        trace_path=str(path),
        rank=rank,
        step=step,
        collective_total_ms=collective_total_ms,
        nccl_kernel_ms=nccl_kernel_ms,
        cuda_stream_sync_ms=cuda_stream_sync_ms,
        unknown_sync_ms=unknown_sync_ms,
        input_dist_pack_ms=input_dist_pack_ms,
        output_dist_unpack_ms=output_dist_unpack_ms,
        unclassified_collective_ms=unclassified_collective_ms,
    )
    return summary.as_row()


def summarize_trace_dir(dir_path: Path) -> list[dict[str, float | str]]:
    if not dir_path.exists():
        return []
    rows = []
    for path in sorted(dir_path.glob("*.pt.trace.json")):
        rows.append(summarize_trace_file(path))
    return rows


def write_trace_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
