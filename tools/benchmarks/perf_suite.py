#!/usr/bin/env python3
"""Run registered performance cases, aggregate them, and check baselines.

The case registry lives in ``tools/benchmarks/perf/cases.json``.  This script is
deliberately a thin layer: it shells out to the existing benchmark binaries and
turn their stdout into machine-readable metrics, so no measurement logic is
duplicated here.

Typical usage::

    # CI / local canary: tiny CPU-only cases, never compares numbers
    python3 tools/benchmarks/perf_suite.py run --label perf_smoke

    # Owned hardware: repeat cases and compare with the per-machine baseline
    python3 tools/benchmarks/perf_suite.py run --label perf --repeats 3 --check

    # Record a new baseline (explicit, reviewed change; never automatic)
    python3 tools/benchmarks/perf_suite.py run --label perf --write-baseline

    # Hot-path PR: interleave two prebuilt trees on the same machine
    python3 tools/benchmarks/perf_suite.py ab --case kvengine_c_read_full \\
        --a-build /tmp/base-build --b-build ./build --repeats 3

Exit codes: 0 ok, 1 execution error, 2 regression against the baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = Path(__file__).resolve().parent / "perf"
CASES_PATH = PERF_DIR / "cases.json"
BASELINE_ROOT = PERF_DIR / "baselines"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results"

CMAKE_CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "ENABLE_CUDA",
    "USE_PERF_REPORT",
    "USE_GPERF_PROFILING",
    "USE_BRPC_CPU_PROFILER",
    "RECSTORE_ENABLE_THIRD_PARTY_ENGINES",
)


class SuiteError(RuntimeError):
    """Raised when a case cannot produce a usable measurement."""


# ── Case registry ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSpec:
    key: str
    unit: str
    direction: str

    @classmethod
    def from_json(cls, raw: dict) -> "MetricSpec":
        direction = raw["direction"]
        if direction not in {"higher", "lower"}:
            raise SuiteError(f"metric direction must be higher|lower: {raw}")
        return cls(key=raw["key"], unit=raw["unit"], direction=direction)


@dataclass(frozen=True)
class Case:
    id: str
    layer: str
    runner: str
    labels: tuple[str, ...]
    repeats: int
    noise_floor_pct: float
    params: dict
    metrics: tuple[MetricSpec, ...]
    description: str = ""

    @classmethod
    def from_json(cls, raw: dict) -> "Case":
        case = cls(
            id=raw["id"],
            layer=raw["layer"],
            runner=raw["runner"],
            labels=tuple(raw.get("labels", ())),
            repeats=int(raw.get("repeats", 1)),
            noise_floor_pct=float(raw["noise_floor_pct"]),
            params=dict(raw.get("params", {})),
            metrics=tuple(MetricSpec.from_json(m) for m in raw["metrics"]),
            description=raw.get("description", ""),
        )
        if case.runner not in RUNNERS:
            raise SuiteError(f"{case.id}: unknown runner {case.runner!r}")
        if not case.metrics:
            raise SuiteError(f"{case.id}: at least one metric is required")
        return case


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise SuiteError(f"unsupported cases schema in {path}")
    cases = [Case.from_json(raw) for raw in payload["cases"]]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise SuiteError(f"duplicate case id: {case.id}")
        seen.add(case.id)
    return cases


def select_cases(
    cases: Iterable[Case], ids: list[str], label: str | None
) -> list[Case]:
    selected = list(cases)
    if ids:
        wanted = set(ids)
        selected = [case for case in selected if case.id in wanted]
        missing = wanted - {case.id for case in selected}
        if missing:
            raise SuiteError(f"unknown case id(s): {', '.join(sorted(missing))}")
    if label:
        selected = [case for case in selected if label in case.labels]
    if not selected:
        raise SuiteError("no case matched the selection")
    return selected


# ── Process helpers ──────────────────────────────────────────────────────────


def _tail(text: str, limit: int = 40) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-limit:])


def _run_command(
    cmd: list[str], log_path: Path, timeout_s: float, cwd: Path = REPO_ROOT
) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        log_path.write_text(_join_cmd(cmd) + "\n\n" + output, encoding="utf-8")
        raise SuiteError(f"command timed out after {timeout_s}s: {cmd[0]}")
    wall_s = time.monotonic() - started
    log_path.write_text(
        _join_cmd(cmd) + f"\n\n{proc.stdout}\nwall_s={wall_s:.3f}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SuiteError(
            f"command failed (exit {proc.returncode}): {cmd[0]}\n"
            f"{_tail(proc.stdout)}"
        )
    return proc.stdout


def _join_cmd(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def _flag(name: str, value: object) -> str:
    if isinstance(value, bool):
        return f"--{name}={'true' if value else 'false'}"
    return f"--{name}={value}"


# ── Runners ──────────────────────────────────────────────────────────────────

_KV_RESULT = re.compile(r"^YCSB_RESULT\b.*throughput_ops_sec=(?P<value>[0-9.eE+-]+)", re.M)
_KV_LATENCY = re.compile(r"^YCSB_LATENCY\b.*p50_us=(?P<p50>[0-9.eE+-]+).*p99_us=(?P<p99>[0-9.eE+-]+)", re.M)
# Tiered cache counters ride on the same YCSB_RESULT line. They are recorded
# opportunistically: a `none` policy prints no counters, so they must never be
# treated as required metrics.
_KV_TIERED = re.compile(r"\b(?P<key>tiered_cache_[a-z_]+)=(?P<value>[0-9.eE+-]+)")
_PS_RESULT = re.compile(
    r"^PS_BENCHMARK_RESULT phase=run\b.*throughput_keys_sec=(?P<value>[0-9.eE+-]+)", re.M
)
_PS_PROFILE = re.compile(
    r"^PS_LOCAL_SHM_PROFILE phase=run\b.*request_total_us_mean=(?P<value>[0-9.eE+-]+)", re.M
)


def _kvengine_work_path(raw: object, work_dir: Path) -> Path:
    """Expand the ``{work_dir}`` placeholder in a per-repeat path parameter."""
    return Path(str(raw).replace("{work_dir}", str(work_dir)))


def _kvengine_command(case: Case, build_dir: Path, work_dir: Path) -> list[str]:
    params = case.params
    cmd = [
        str(build_dir / "bin" / "benchmark_kv_engine"),
        _flag("dram_path", work_dir / "dram"),
        _flag("index_type", params["index_type"]),
        _flag("value_store_type", params["value_store_type"]),
        _flag("workload", params["workload"]),
        _flag("read_mode", params["read_mode"]),
        _flag("distribution", params["distribution"]),
        _flag("record_count", params["record_count"]),
        _flag("batch_keys", params["batch_keys"]),
        _flag("thread_num", params["thread_num"]),
        _flag("load_thread_num", params["thread_num"]),
        _flag("running_seconds", params["running_seconds"]),
        _flag("value_size", params["value_size"]),
        _flag("report_latency", params.get("report_latency", False)),
    ]
    if params.get("dram_capacity_bytes"):
        cmd.append(_flag("dram_capacity_bytes", params["dram_capacity_bytes"]))
    if "zipfian_alpha" in params:
        cmd.append(_flag("zipfian_alpha", params["zipfian_alpha"]))
    if "zipfian_scramble" in params:
        cmd.append(_flag("zipfian_scramble", bool(params["zipfian_scramble"])))
    if params.get("ssd_path"):
        cmd.append(_flag("ssd_path", _kvengine_work_path(params["ssd_path"], work_dir)))
    if params.get("ssd_capacity_bytes"):
        cmd.append(_flag("ssd_capacity_bytes", params["ssd_capacity_bytes"]))
    if params.get("ssd_io_backend"):
        cmd.append(_flag("ssd_io_backend", params["ssd_io_backend"]))
    if params.get("ssd_queue_depth"):
        cmd.append(_flag("ssd_queue_depth", params["ssd_queue_depth"]))
    for key in (
        "tiering_cache_policy",
        "tiering_cache_budget_ratio",
        "tiering_cache_shards",
        "tiering_promote_min_misses",
    ):
        if key in params:
            cmd.append(_flag(key, params[key]))
    return cmd


def _result_line(output: str, match: "re.Match[str]") -> str:
    """Return the full text of the line the match starts on."""
    end = output.find("\n", match.start())
    return output[match.start() :] if end < 0 else output[match.start() : end]

def _parse_kvengine(output: str) -> dict[str, float]:
    match = _KV_RESULT.search(output)
    if match is None:
        raise SuiteError("benchmark_kv_engine printed no YCSB_RESULT line")
    metrics = {"throughput_ops_sec": float(match.group("value"))}
    latency = _KV_LATENCY.search(output)
    if latency is not None:
        metrics["p50_us"] = float(latency.group("p50"))
        metrics["p99_us"] = float(latency.group("p99"))
    for tiered in _KV_TIERED.finditer(_result_line(output, match)):
        metrics[tiered.group("key")] = float(tiered.group("value"))
    return metrics


def _clean_within(work_dir: Path, target: Path) -> None:
    """Delete ``target`` only when it lives inside ``work_dir``.

    A stale tiered run leaves its SSD files behind; reusing them would change the
    residency/hit pattern of the next repeat.
    """
    try:
        target.resolve().relative_to(work_dir.resolve())
    except ValueError as exc:
        raise SuiteError(f"refusing to clean {target} outside {work_dir}") from exc
    if target.resolve() == work_dir.resolve():
        return
    shutil.rmtree(target, ignore_errors=True)


def _prepare_work_dir(case: Case, work_dir: Path) -> None:
    """Reset the per-repeat scratch tree.

    The tiered SSD tree is deleted explicitly before the run (a leftover file
    would otherwise change the residency pattern) and recreated because
    ``benchmark_kv_engine`` opens its SSD file without creating the parent
    directory.
    """
    shutil.rmtree(work_dir, ignore_errors=True)
    (work_dir / "dram").mkdir(parents=True, exist_ok=True)
    ssd_path = case.params.get("ssd_path")
    if ssd_path:
        target = _kvengine_work_path(ssd_path, work_dir)
        _clean_within(work_dir, target)
        target.mkdir(parents=True, exist_ok=True)


def _run_kvengine(
    case: Case, build_dir: Path, output_dir: Path, repeat: int, timeout_s: float
) -> dict[str, float]:
    work_dir = output_dir / "work" / case.id / f"r{repeat:02d}"
    _prepare_work_dir(case, work_dir)
    log_path = output_dir / "logs" / case.id / f"r{repeat:02d}.log"
    output = _run_command(
        _kvengine_command(case, build_dir, work_dir), log_path, timeout_s
    )
    return _parse_kvengine(output)


def _local_shm_config(case: Case, region: str, device_dir: Path) -> dict:
    params = case.params
    value_size = int(params["value_size"])
    capacity = int(params["capacity"])
    return {
        "cache_ps": {
            "ps_type": "LOCAL_SHM",
            "num_threads": 1,
            "base_kv_config": {
                "capacity": capacity,
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "path": str(device_dir / "kv_store" / "value"),
                    "default_value_size_hint": value_size,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": capacity * value_size,
                    },
                },
            },
        },
        "local_shm": {
            "region_name": region,
            "slot_count": int(params["slot_count"]),
            "ready_queue_count": 1,
            "ready_queue_burst_limit": 8,
            "slot_buffer_bytes": int(params["slot_buffer_bytes"]),
            "client_timeout_ms": 30000,
        },
    }


def _wait_for_region(region: str, timeout_s: float, proc: subprocess.Popen) -> None:
    shm_path = Path("/dev/shm") / region
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SuiteError(f"local_shm_ps_server exited early ({proc.returncode})")
        if shm_path.exists():
            # Give the server one scheduling slice to finish publishing the
            # control block after the region file appears.
            time.sleep(0.2)
            return
        time.sleep(0.05)
    raise SuiteError(f"local_shm region {region!r} did not appear within {timeout_s}s")


def _run_ps_local_shm(
    case: Case, build_dir: Path, output_dir: Path, repeat: int, timeout_s: float
) -> dict[str, float]:
    params = case.params
    run_id = f"{int(time.time())}_{repeat:02d}_{uuid.uuid4().hex[:6]}"
    region = f"perf_{run_id}"
    work_dir = output_dir / "work" / case.id / f"r{repeat:02d}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "config.json"
    config_path.write_text(
        json.dumps(_local_shm_config(case, region, work_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    shm_path = Path("/dev/shm") / region
    server_log = output_dir / "logs" / case.id / f"r{repeat:02d}.server.log"
    client_log = output_dir / "logs" / case.id / f"r{repeat:02d}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)

    shm_path.unlink(missing_ok=True)
    with server_log.open("w", encoding="utf-8") as sink:
        server = subprocess.Popen(
            [
                str(build_dir / "bin" / "local_shm_ps_server"),
                _flag("config_path", config_path),
            ],
            cwd=str(REPO_ROOT),
            stdout=sink,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        _wait_for_region(region, float(params.get("startup_timeout_s", 10.0)), server)
        cmd = [
            str(build_dir / "bin" / "ps_transport_benchmark"),
            _flag("transport", "local_shm"),
            _flag("config_path", config_path),
            _flag("workload", "transactions"),
            _flag("mode", "fetch"),
            _flag("record_count", params["record_count"]),
            _flag("batch_keys", params["batch_keys"]),
            _flag("thread_num", params["thread_num"]),
            _flag("load_thread_num", params["thread_num"]),
            _flag("running_seconds", params["running_seconds"]),
            _flag("value_size", params["value_size"]),
            _flag("report_mode", "summary"),
        ]
        output = _run_command(cmd, client_log, timeout_s)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        shm_path.unlink(missing_ok=True)

    metrics = {"throughput_keys_sec": _parse_float(_PS_RESULT, output, "PS_BENCHMARK_RESULT")}
    profile = _PS_PROFILE.search(output)
    if profile is not None:
        metrics["request_total_us"] = float(profile.group("value"))
    return metrics


def _parse_float(pattern: re.Pattern, output: str, label: str) -> float:
    match = pattern.search(output)
    if match is None:
        raise SuiteError(f"benchmark printed no {label} summary line")
    return float(match.group("value"))


Runner = Callable[[Case, Path, Path, int, float], dict[str, float]]

RUNNERS: dict[str, Runner] = {
    "kvengine": _run_kvengine,
    "ps_local_shm": _run_ps_local_shm,
}


# ── Environment fingerprint ──────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_mhz_max() -> float:
    """Highest reported core clock, i.e. the current boost ceiling."""
    try:
        freqs = [
            float(line.split(":", 1)[1])
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("cpu MHz")
        ]
    except (OSError, ValueError):
        return 0.0
    return round(max(freqs), 1) if freqs else 0.0


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sluggify(value: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:limit] or "unknown"


def default_env_id() -> str:
    return f"{_sluggify(socket.gethostname(), 24)}-{_sluggify(_cpu_model())}"


def collect_env(
    build_dir: Path, binaries: list[Path], env_id: str | None = None
) -> dict:
    cache: dict[str, str] = {}
    cache_path = build_dir / "CMakeCache.txt"
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line or line.startswith("//"):
                continue
            key, _, value = line.partition("=")
            name = key.split(":", 1)[0]
            if name in CMAKE_CACHE_KEYS:
                cache[name] = value
    infiniband = ""
    try:
        infiniband = ",".join(sorted(os.listdir("/sys/class/infiniband")))
    except OSError:
        pass
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "env_id": env_id or default_env_id(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "loadavg_start": os.getloadavg(),
        "cpu_mhz_start": _cpu_mhz_max(),
        "python": platform.python_version(),
        "infiniband_devices": infiniband,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "build_dir": str(build_dir),
        "cmake_cache": cache,
        "binaries": {
            str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path): {
                "sha256": _sha256(path),
                "mtime": (
                    datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                        timespec="seconds"
                    )
                    if path.exists()
                    else "missing"
                ),
            }
            for path in binaries
        },
    }


def _case_binaries(case: Case, build_dir: Path) -> list[Path]:
    if case.runner == "kvengine":
        return [build_dir / "bin" / "benchmark_kv_engine"]
    if case.runner == "ps_local_shm":
        return [
            build_dir / "bin" / "local_shm_ps_server",
            build_dir / "bin" / "ps_transport_benchmark",
        ]
    return []


# ── Aggregation and evaluation ───────────────────────────────────────────────


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _spread_pct(values: list[float]) -> float:
    """Peak-to-peak spread relative to the median, in percent."""
    mid = median(values)
    if not mid or len(values) < 2:
        return 0.0
    return (max(values) - min(values)) / mid * 100.0


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["case"], row["metric"]), []).append(row["value"])
    summary = []
    for (case_id, metric_key), values in grouped.items():
        summary.append(
            {
                "case": case_id,
                "metric": metric_key,
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "spread_pct": round(_spread_pct(values), 3),
                "samples": len(values),
            }
        )
    summary.sort(key=lambda item: (item["case"], item["metric"]))
    return summary


def evaluate(
    case: Case,
    metric: MetricSpec,
    observed: dict,
    baseline: dict | None,
    floor_pct: float,
    env: dict | None = None,
    clock_tolerance: float = 0.15,
    baseline_mismatch: str | None = None,
) -> dict:
    row = dict(observed)
    row["unit"] = metric.unit
    row["direction"] = metric.direction
    row["threshold_pct"] = floor_pct
    unstable = (
        float(observed.get("spread_pct") or 0.0) > floor_pct
        and int(observed.get("samples") or 0) > 1
    )
    if baseline_mismatch is not None:
        # The recorded numbers were measured under other conditions; gating on
        # them would report a config change as a code regression.
        row["baseline"] = None
        row["ratio"] = None
        row["status"] = "baseline-mismatch"
        row["mismatch"] = baseline_mismatch
        return row
    if baseline is None:
        row["baseline"] = None
        row["ratio"] = None
        row["status"] = "unstable" if unstable else "no-baseline"
        return row
    base = float(baseline["median"])
    row["baseline"] = base
    base_mhz = baseline.get("cpu_mhz_max")
    current_mhz = (env or {}).get("cpu_mhz_max")
    if base_mhz and current_mhz:
        if abs(float(current_mhz) / float(base_mhz) - 1.0) > clock_tolerance:
            row["ratio"] = None
            row["status"] = "env-mismatch"
            return row
    if base == 0:
        row["ratio"] = None
        row["status"] = "no-baseline"
        return row
    if unstable:
        # The run's own repeats disagree by more than the noise floor, so the
        # median is not a measurement yet: report the ratio but never gate on it.
        row["ratio"] = observed["median"] / base
        row["status"] = "unstable"
        return row
    ratio = observed["median"] / base
    row["ratio"] = ratio
    limit = floor_pct / 100.0
    if metric.direction == "higher":
        regressed = ratio < 1.0 - limit
    else:
        regressed = ratio > 1.0 + limit
    row["status"] = "regressed" if regressed else "ok"
    return row


def baseline_dir(root: Path | None = None) -> Path:
    return root if root is not None else BASELINE_ROOT


def baseline_mismatch(baseline: dict, case: Case, env: dict) -> str | None:
    """Why ``baseline`` cannot be compared with this run, or None if it can.

    Values are only comparable under the same workload and the same build
    knobs. Re-registering a case under the same id with other parameters, or
    rebuilding with a different cmake cache, would otherwise surface as a code
    regression.
    """
    if (baseline.get("resolution_hint") or {}) != case.params:
        return "workload parameters changed"
    if (baseline.get("cmake_cache") or {}) != (env.get("cmake_cache") or {}):
        return "cmake cache changed"
    return None


def load_baseline(
    env_id: str, case_id: str, root: Path | None = None
) -> dict | None:
    path = baseline_dir(root) / env_id / f"{case_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(
    env_id: str,
    case: Case,
    summary: list[dict],
    env: dict,
    root: Path | None = None,
) -> Path:
    metrics = {
        item["metric"]: {
            "median": item["median"],
            "min": item["min"],
            "max": item["max"],
            "spread_pct": item["spread_pct"],
            "samples": item["samples"],
        }
        for item in summary
        if item["case"] == case.id
    }
    payload = {
        "schema": 1,
        "case": case.id,
        "env_id": env_id,
        "commit": env.get("git_commit", ""),
        "created": env.get("created", ""),
        "repeats": case.repeats,
        "resolution_hint": case.params,
        "cpu_mhz_max": env.get("cpu_mhz_max"),
        "cmake_cache": env.get("cmake_cache", {}),
        "binaries": {
            name: info.get("sha256", "missing")
            for name, info in (env.get("binaries") or {}).items()
        },
        "metrics": metrics,
    }
    path = baseline_dir(root) / env_id / f"{case.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ── Rendering ────────────────────────────────────────────────────────────────


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.3f}"


def render_summary(
    *,
    title: str,
    rows: list[dict],
    env: dict,
    baseline_env_id: str,
    check: bool,
    missing_baselines: list[str] | None = None,
) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- host: `{env['hostname']}` (env bucket `{baseline_env_id}`)")
    lines.append(f"- commit: `{env.get('git_commit', '')[:12]}` dirty={env.get('git_dirty')}")
    lines.append(f"- build: `{env.get('build_dir')}`")
    cache = env.get("cmake_cache", {})
    if cache:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(cache.items()))
        lines.append(f"- cmake: {rendered}")
    lines.append(f"- loadavg at start: {tuple(round(v, 2) for v in env.get('loadavg_start', ()))}")
    lines.append("")
    header = ["case", "metric", "median", "spread%", "baseline", "ratio", "status"]
    table = [header]
    for row in rows:
        table.append(
            [
                row["case"],
                f"{row['metric']} ({row['unit']})",
                _fmt(row["median"]),
                f"{row['spread_pct']:.1f}",
                _fmt(row.get("baseline")) if check else "-",
                f"{row['ratio']:.3f}" if check and row.get("ratio") else "-",
                row.get("status", "-") if check else "-",
            ]
        )
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    lines.append("| " + " | ".join(header[i].ljust(widths[i]) for i in range(len(header))) + " |")
    lines.append("|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|")
    for line in table[1:]:
        lines.append("| " + " | ".join(line[i].ljust(widths[i]) for i in range(len(line))) + " |")
    regressions = [row for row in rows if row.get("status") == "regressed"]
    lines.append("")
    if missing_baselines:
        lines.append(
            "No baseline for: "
            + ", ".join(f"`{case_id}`" for case_id in missing_baselines)
            + " (record one with `--write-baseline` on this machine)."
        )
    mismatches = [row for row in rows if row.get("status") == "env-mismatch"]
    if mismatches:
        lines.append(
            "## Environment mismatch\n"
            "These rows were not compared: the CPU clock ceiling of this run does not "
            "match the baseline (re-record the baseline or free the machine)."
        )
        for row in mismatches:
            lines.append(f"- `{row['case']}` {row['metric']}")
        lines.append("")
    unstable = [row for row in rows if row.get("status") == "unstable"]
    stale_baselines = [row for row in rows if row.get("status") == "baseline-mismatch"]
    if stale_baselines:
        lines.append(
            "## Baseline mismatch (not compared)\n"
            "The stored baseline was recorded with other workload parameters or "
            "build knobs, so it is not a reference for this run. Re-record it with "
            "`--write-baseline` after confirming the new settings."
        )
        for row in stale_baselines:
            lines.append(f"- `{row['case']}` {row['metric']}: {row.get('mismatch', '')}")
        lines.append("")
    if unstable:
        lines.append(
            "## Unstable (not gated)\n"
            "The repeats of these rows disagree by more than the noise floor, so the "
            "median is not a measurement yet. Re-run on an idle machine, or compare "
            "two configs with `perf_suite.py ab` instead of a baseline."
        )
        for row in unstable:
            lines.append(
                f"- `{row['case']}` {row['metric']}: spread {row['spread_pct']:.1f}% "
                f"> floor {row['threshold_pct']}% (median {_fmt(row['median'])})"
            )
        lines.append("")
    if regressions:
        lines.append("## Regressions")
        for row in regressions:
            lines.append(
                f"- `{row['case']}` {row['metric']}: {_fmt(row['median'])} vs baseline "
                f"{_fmt(row['baseline'])} (ratio {row['ratio']:.3f}, "
                f"threshold {row['threshold_pct']}%)"
            )
    elif check:
        lines.append("No regression beyond the per-case noise floor.")
    return "\n".join(lines) + "\n"


# ── Commands ─────────────────────────────────────────────────────────────────


def _resolve_output_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else REPO_ROOT / path
    stamp = datetime.now().strftime("%m%d%H%M")
    candidate = DEFAULT_OUTPUT_ROOT / f"perf_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = DEFAULT_OUTPUT_ROOT / f"perf_{stamp}_{suffix}"
        suffix += 1
    return candidate


def _resolve_baseline_dir(raw: str | None) -> Path:
    path = Path(raw) if raw else BASELINE_ROOT
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_build_dir(raw: str | None) -> Path:
    path = Path(raw) if raw else DEFAULT_BUILD_DIR
    return path if path.is_absolute() else REPO_ROOT / path


def cmd_list(args: argparse.Namespace) -> int:
    cases = select_cases(load_cases(), args.case, args.label)
    header = ["case", "layer", "labels", "repeats", "floor%", "metrics"]
    table = [header]
    for case in cases:
        table.append(
            [
                case.id,
                case.layer,
                ",".join(case.labels),
                str(case.repeats),
                f"{case.noise_floor_pct:g}",
                ",".join(f"{m.key}({m.direction})" for m in case.metrics),
            ]
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    return 0


def _reset_failure_marker(output_dir: Path) -> None:
    """Forget a previous run's failures so a fixed rerun exits clean.

    The exit status of ``run`` is derived from the presence of this file, and
    CTest reuses one output directory per smoke case, so a stale marker turns
    every later successful run into a failure.
    """
    marker = output_dir / "failures.txt"
    if marker.exists():
        marker.unlink()


def run_cases(
    cases: list[Case],
    build_dir: Path,
    output_dir: Path,
    repeats_override: int | None,
    timeout_s: float,
    label: str,
    env_id: str | None = None,
) -> tuple[list[dict], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_failure_marker(output_dir)
    binaries = sorted({path for case in cases for path in _case_binaries(case, build_dir)})
    env = collect_env(build_dir, binaries, env_id)
    env["label"] = label
    (output_dir / "env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows: list[dict] = []
    failures: list[str] = []
    for case in cases:
        repeats = repeats_override or case.repeats
        print(f"[perf] {case.id}: {repeats} repeat(s) via {case.runner}", flush=True)
        runner = RUNNERS[case.runner]
        for repeat in range(repeats):
            try:
                metrics = runner(case, build_dir, output_dir, repeat, timeout_s)
            except SuiteError as exc:
                failures.append(f"{case.id} repeat {repeat}: {exc}")
                print(f"[perf]   repeat {repeat}: FAILED ({exc})", flush=True)
                continue
            missing = [metric.key for metric in case.metrics if metric.key not in metrics]
            if missing:
                failures.append(
                    f"{case.id} repeat {repeat}: missing metric(s) {missing}"
                )
                continue
            for metric in case.metrics:
                rows.append(
                    {
                        "case": case.id,
                        "layer": case.layer,
                        "runner": case.runner,
                        "metric": metric.key,
                        "unit": metric.unit,
                        "direction": metric.direction,
                        "repeat": repeat,
                        "value": metrics[metric.key],
                    }
                )
            env.setdefault("cpu_mhz_samples", []).append(_cpu_mhz_max())
            print(
                "  repeat {}: {}".format(
                    repeat,
                    ", ".join(f"{m.key}={metrics[m.key]:.4g}" for m in case.metrics),
                ),
                flush=True,
            )
    env["loadavg_end"] = os.getloadavg()
    samples = env.get("cpu_mhz_samples") or []
    env["cpu_mhz_max"] = max(samples) if samples else _cpu_mhz_max()
    (output_dir / "env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        (output_dir / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    return rows, env


def cmd_run(args: argparse.Namespace) -> int:
    cases = select_cases(load_cases(), args.case, args.label)
    build_dir = _resolve_build_dir(args.build_dir)
    output_dir = _resolve_output_dir(args.output_dir)
    env_id = args.env_id or default_env_id()
    rows, env = run_cases(
        cases,
        build_dir,
        output_dir,
        args.repeats,
        args.timeout,
        args.label or "all",
        env_id=env_id,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps({"schema": 1, "env": env, "rows": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if not rows:
        print("[perf] no metrics collected; see failures.txt", file=sys.stderr)
        return 1

    summary = summarize(rows)
    evaluated = []
    missing_baselines: list[str] = []
    for case in cases:
        case_rows = [item for item in summary if item["case"] == case.id]
        if not case_rows:
            continue
        baseline = (
            load_baseline(env_id, case.id, _resolve_baseline_dir(args.baseline_dir))
            if args.check
            else None
        )
        mismatch = (
            baseline_mismatch(baseline, case, env) if baseline is not None else None
        )
        if args.check and baseline is None:
            missing_baselines.append(case.id)
        baseline_metrics = (baseline or {}).get("metrics")
        metrics_by_key = {m.key: m for m in case.metrics}
        for item in case_rows:
            metric = metrics_by_key[item["metric"]]
            floor = (
                args.noise_floor_pct
                if args.noise_floor_pct is not None
                else case.noise_floor_pct
            )
            base_entry = (
                baseline_metrics.get(item["metric"]) if baseline_metrics else None
            )
            if base_entry is not None and baseline.get("cpu_mhz_max"):
                base_entry = dict(base_entry, cpu_mhz_max=baseline["cpu_mhz_max"])
            evaluated.append(
                evaluate(
                    case,
                    metric,
                    item,
                    base_entry,
                    floor,
                    env=env,
                    baseline_mismatch=mismatch,
                )
            )
    summary = evaluated

    report = render_summary(
        title=f"perf suite: {args.label or 'all cases'}",
        rows=summary,
        env=env,
        baseline_env_id=env_id,
        check=args.check,
        missing_baselines=missing_baselines,
    )
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)

    if args.write_baseline:
        if not args.check:
            root = _resolve_baseline_dir(args.baseline_dir)
            unstable_cases = sorted(
                {row["case"] for row in summary if row.get("status") == "unstable"}
            )
            for case in cases:
                write_baseline(env_id, case, summary, env, root)
            print(f"[perf] baselines written under {root / env_id}")
            if unstable_cases:
                print(
                    "[perf] WARNING: unstable rows recorded anyway: "
                    + ", ".join(unstable_cases)
                    + ". A noisy median is a snapshot, not a reference; re-run on an "
                    "idle machine or gate these cases with `ab`.",
                    file=sys.stderr,
                )
        else:
            print("[perf] --write-baseline is ignored together with --check", file=sys.stderr)

    if args.check and any(row.get("status") == "regressed" for row in summary):
        return 2
    return 0 if not (output_dir / "failures.txt").exists() else 1


def _apply_param_overrides(case: Case, raw: list[str]) -> Case:
    """Return the case with ``--a-param k=v`` style overrides applied.

    Values are parsed as JSON when possible, so ``zipfian_scramble=true`` becomes
    a bool and ``record_count=50000`` an int; anything else stays a string.
    """
    if not raw:
        return case
    params = dict(case.params)
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SuiteError(f"param override must be key=value: {item!r}")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return replace(case, params=params)


def ab_status(
    direction: str, a_values: list[float], b_values: list[float], floor_pct: float
) -> tuple[float, str]:
    """Decide an A/B row from the two sample sets.

    Returns ``(b_over_a, status)``. Repeats are interleaved inside one session, so
    a difference counts when the two sample ranges are disjoint: that holds even
    when each side is noisy, and it keeps a 3x effect reportable at 15% spread
    while a 6% wobble inside the floor stays inconclusive.
    """
    a_med, b_med = median(a_values), median(b_values)
    ratio = b_med / a_med if a_med else float("nan")
    limit = floor_pct / 100.0
    regressed = ratio < 1.0 - limit if direction == "higher" else ratio > 1.0 + limit
    overlap = max(a_values) >= min(b_values) and max(b_values) >= min(a_values)
    if regressed and overlap:
        return ratio, "unstable"
    return ratio, ("regressed" if regressed else "ok")


def cmd_ab(args: argparse.Namespace) -> int:
    cases = select_cases(load_cases(), [args.case], None)
    base_case = cases[0]
    case_a = _apply_param_overrides(base_case, args.a_param)
    case_b = _apply_param_overrides(base_case, args.b_param)
    a_dir = _resolve_build_dir(args.a_build)
    b_dir = _resolve_build_dir(args.b_build)
    output_dir = _resolve_output_dir(args.output_dir)
    repeats = args.repeats or base_case.repeats
    runner = RUNNERS[base_case.runner]

    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_failure_marker(output_dir)
    env = collect_env(
        b_dir, sorted(set(_case_binaries(base_case, a_dir) + _case_binaries(base_case, b_dir)))
    )
    env["ab"] = {
        "case": base_case.id,
        "a_build": str(a_dir),
        "b_build": str(b_dir),
        "a_params": case_a.params,
        "b_params": case_b.params,
        "repeats": repeats,
    }
    (output_dir / "env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    samples: dict[str, dict[str, list[float]]] = {"a": {}, "b": {}}
    failures: list[str] = []
    # Interleave A/B/A/B so machine drift cannot favour one side.
    for repeat in range(repeats):
        for variant, build, case in (("a", a_dir, case_a), ("b", b_dir, case_b)):
            try:
                metrics = runner(case, build, output_dir / variant, repeat, args.timeout)
            except SuiteError as exc:
                failures.append(f"{variant} repeat {repeat}: {exc}")
                print(f"[perf] {variant} repeat {repeat}: FAILED ({exc})", flush=True)
                continue
            for metric in case.metrics:
                if metric.key in metrics:
                    samples[variant].setdefault(metric.key, []).append(metrics[metric.key])
            declared = {metric.key for metric in case.metrics}
            extras = {key: value for key, value in metrics.items() if key not in declared}
            print(
                f"[perf] {variant} repeat {repeat}: "
                + ", ".join(f"{m.key}={metrics.get(m.key, float('nan')):.4g}" for m in case.metrics)
                + "".join(f", {key}={value:.4g}" for key, value in sorted(extras.items())),
                flush=True,
            )
    if failures:
        (output_dir / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")

    floor_pct = base_case.noise_floor_pct if args.noise_floor_pct is None else args.noise_floor_pct
    rows = []
    for metric in base_case.metrics:
        a_values = samples["a"].get(metric.key, [])
        b_values = samples["b"].get(metric.key, [])
        if not a_values or not b_values:
            continue
        a_med, b_med = median(a_values), median(b_values)
        a_spread = _spread_pct(a_values)
        b_spread = _spread_pct(b_values)
        ratio, status = ab_status(metric.direction, a_values, b_values, floor_pct)
        rows.append(
            {
                "case": base_case.id,
                "metric": metric.key,
                "unit": metric.unit,
                "a_median": a_med,
                "b_median": b_med,
                "a_spread_pct": round(a_spread, 3),
                "b_spread_pct": round(b_spread, 3),
                "ratio": ratio,
                "status": status,
            }
        )
    lines = [f"# A/B: {base_case.id}", ""]
    lines.append(f"- A build: `{a_dir}`")
    lines.append(f"- B build: `{b_dir}`")
    for variant, case in (("A", case_a), ("B", case_b)):
        diff = {
            key: value
            for key, value in case.params.items()
            if base_case.params.get(key) != value
        }
        if diff:
            rendered = ", ".join(f"{key}={value}" for key, value in sorted(diff.items()))
            lines.append(f"- {variant} overrides: {rendered}")
    lines.append(f"- repeats: {repeats} (interleaved A/B)")
    lines.append("")
    lines.append("| metric | A median | A spread% | B median | B spread% | B/A | status |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['metric']} ({row['unit']}) | {_fmt(row['a_median'])} | "
            f"{row['a_spread_pct']:.1f} | {_fmt(row['b_median'])} | "
            f"{row['b_spread_pct']:.1f} | {row['ratio']:.3f} | {row['status']} |"
        )
    unstable = [row for row in rows if row["status"] == "unstable"]
    if unstable:
        lines.append("")
        lines.append(
            "Inconclusive: the apparent change is within the floor and the A/B sample "
            "ranges overlap, so it cannot be separated from run-to-run noise. Increase "
            "`--repeats` or make the case less noisy before quoting the ratio."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "ab.json").write_text(
        json.dumps({"schema": 1, "case": base_case.id, "rows": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    if args.check and any(row["status"] == "regressed" for row in rows):
        return 2
    return 0 if not (output_dir / "failures.txt").exists() else 1


def cmd_aggregate(args: argparse.Namespace) -> int:
    root = Path(args.input) if args.input else DEFAULT_OUTPUT_ROOT
    root = root if root.is_absolute() else REPO_ROOT / root
    output = Path(args.csv) if args.csv else root / "perf_history.csv"
    output = output if output.is_absolute() else REPO_ROOT / output
    rows: list[dict] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        env = payload.get("env", {})
        for row in payload.get("rows", []):
            rows.append(
                {
                    "run_dir": str(metrics_path.parent),
                    "created": env.get("created", ""),
                    "env_id": env.get("env_id", ""),
                    "commit": env.get("git_commit", "")[:12],
                    "dirty": env.get("git_dirty", ""),
                    "case": row.get("case", ""),
                    "layer": row.get("layer", ""),
                    "metric": row.get("metric", ""),
                    "unit": row.get("unit", ""),
                    "repeat": row.get("repeat", ""),
                    "value": row.get("value", ""),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_dir",
                "created",
                "env_id",
                "commit",
                "dirty",
                "case",
                "layer",
                "metric",
                "unit",
                "repeat",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[perf] wrote {len(rows)} row(s) to {output}")
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    build_dir = _resolve_build_dir(args.build_dir)
    env = collect_env(build_dir, [build_dir / "bin" / "benchmark_kv_engine"])
    print(json.dumps(env, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_selection(target: argparse.ArgumentParser) -> None:
        target.add_argument("--case", action="append", default=[], help="case id (repeatable)")
        target.add_argument("--label", help="only cases carrying this label")

    list_parser = sub.add_parser("list", help="list registered cases")
    add_selection(list_parser)
    list_parser.set_defaults(func=cmd_list)

    run_parser = sub.add_parser("run", help="run cases and write metrics.json/summary.md")
    add_selection(run_parser)
    run_parser.add_argument("--build-dir", help="build tree containing bin/ (default: ./build)")
    run_parser.add_argument("--output-dir", help="run directory (default: results/perf_<stamp>)")
    run_parser.add_argument("--repeats", type=int, help="override the case repeat count")
    run_parser.add_argument("--timeout", type=float, default=300.0, help="per-run timeout seconds")
    run_parser.add_argument("--env-id", help="baseline bucket (default: hostname-cpu)")
    run_parser.add_argument(
        "--baseline-dir", help="baseline root (default: tools/benchmarks/perf/baselines)"
    )
    run_parser.add_argument("--check", action="store_true", help="compare with the baseline")
    run_parser.add_argument(
        "--noise-floor-pct", type=float, help="override the per-case noise floor"
    )
    run_parser.add_argument(
        "--write-baseline", action="store_true", help="store the medians as the new baseline"
    )
    run_parser.set_defaults(func=cmd_run)

    ab_parser = sub.add_parser(
        "ab", help="interleaved A/B of two build trees or two configs"
    )
    ab_parser.add_argument("--case", required=True)
    ab_parser.add_argument(
        "--a-build", help="build tree for side A (default: ./build)"
    )
    ab_parser.add_argument(
        "--b-build", help="build tree for side B (default: ./build)"
    )
    ab_parser.add_argument(
        "--a-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a case param for side A (repeatable, JSON values)",
    )
    ab_parser.add_argument(
        "--b-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a case param for side B (repeatable, JSON values)",
    )
    ab_parser.add_argument("--repeats", type=int)
    ab_parser.add_argument("--output-dir")
    ab_parser.add_argument("--timeout", type=float, default=600.0)
    ab_parser.add_argument("--noise-floor-pct", type=float)
    ab_parser.add_argument("--check", action="store_true", help="exit 2 when B regresses")
    ab_parser.set_defaults(func=cmd_ab)

    agg_parser = sub.add_parser("aggregate", help="collect metrics.json files into one CSV")
    agg_parser.add_argument("--input", help="root scanned for metrics.json (default: results/)")
    agg_parser.add_argument("--csv", help="output CSV path")
    agg_parser.set_defaults(func=cmd_aggregate)

    env_parser = sub.add_parser("env", help="print the environment fingerprint as JSON")
    env_parser.add_argument("--build-dir")
    env_parser.set_defaults(func=cmd_env)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SuiteError as exc:
        print(f"[perf] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
