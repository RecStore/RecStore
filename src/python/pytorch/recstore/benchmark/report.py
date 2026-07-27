from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


def setup_local_report_env(jsonl_path: str) -> None:
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    open(jsonl_path, "w", encoding="utf-8").close()
    os.environ["RECSTORE_REPORT_MODE"] = "local"
    os.environ["RECSTORE_REPORT_LOCAL_SINK"] = "jsonl"
    os.environ["RECSTORE_REPORT_JSONL_PATH"] = jsonl_path
    os.environ.setdefault("RECSTORE_REPORT_FLUSH_EVERY_N", "256")
    os.environ.setdefault("RECSTORE_LOCAL_SHM_STAGE_REPORT", "1")


def summarize_us(values: list[float]) -> str:
    if not values:
        return "count=0"
    s = sorted(values)
    p50 = s[len(s) // 2]
    p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
    return (
        f"count={len(values)} mean={statistics.fmean(values):.2f}us "
        f"p50={p50:.2f}us p95={p95:.2f}us max={s[-1]:.2f}us"
    )


def analyze_stage_table(
    repo_root: Path,
    jsonl_path: str,
    csv_path: str,
    table_name: str,
    top_n: int = 20,
    extra_inputs: list[str] | None = None,
) -> str:
    cmd = [
        sys.executable,
        str(repo_root / "src/test/scripts/analyze_embupdate_stages.py"),
        "--input",
        jsonl_path,
    ]
    for path in extra_inputs or []:
        if path:
            cmd.extend(["--input", path])
    cmd.extend(
        [
            "--group-by-prefix",
            "--table-name",
            table_name,
            "--export-csv",
            csv_path,
            "--top",
            str(top_n),
        ]
    )
    res = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"analyze failed: {res.stderr}")
    return res.stdout


def analyze_embupdate(
    repo_root: Path,
    jsonl_path: str,
    csv_path: str,
    top_n: int = 20,
    extra_inputs: list[str] | None = None,
) -> str:
    return analyze_stage_table(
        repo_root=repo_root,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        table_name="embupdate_stages",
        top_n=top_n,
        extra_inputs=extra_inputs,
    )


def write_stage_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalize_torchrec_row(row: dict) -> dict:
    row["collective_total_ms"] = row["collective_launch_ms"] + row["collective_wait_ms"]
    row["embed_transport_ms"] = row["collective_total_ms"]
    row["network_proxy_torchrec_ms"] = row["collective_total_ms"]
    row["kv_local_only_ms"] = row["embed_lookup_local_ms"] + row["embed_pool_local_ms"]
    row["kv_extended_ms"] = (
        row["input_pack_ms"]
        + row["embed_lookup_local_ms"]
        + row["embed_pool_local_ms"]
        + row["output_unpack_ms"]
    )
    row["emb_stage_ms"] = row["kv_extended_ms"]
    row["network_proxy_torchrec_extended_ms"] = (
        row["collective_total_ms"] + row["input_pack_ms"] + row["output_unpack_ms"]
    )
    return row


def finalize_recstore_row(row: dict) -> dict:
    if (
        "local_update_backend_call_ms" not in row
        and "local_update_shm_call_ms" in row
    ):
        row["local_update_backend_call_ms"] = float(
            row.get("local_update_shm_call_ms", 0.0)
        )
    row["emb_stage_ms"] = (
        row["input_pack_ms"]
        + row["embed_lookup_local_ms"]
        + row["embed_pool_local_ms"]
        + row["output_unpack_ms"]
    )
    row["lookup_breakdown_ms"] = (
        float(row.get("lookup_wait_ms", 0.0))
        + float(row.get("lookup_owner_exchange_ms", 0.0))
        + float(row.get("lookup_local_lookup_ms", 0.0))
        + float(row.get("lookup_reassemble_ms", 0.0))
    )
    row["sparse_update_breakdown_ms"] = (
        float(row.get("update_trace_merge_ms", 0.0))
        + float(row.get("update_owner_exchange_ms", 0.0))
        + float(row.get("update_local_apply_ms", 0.0))
        + float(row.get("update_flush_wait_ms", 0.0))
    )
    return row
