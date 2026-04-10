from __future__ import annotations

import csv
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    measured = [row for row in rows if row.get("warmup_excluded", "") != "1"]
    return measured or rows


def _mean(rows: list[dict[str, str]], column: str) -> float | None:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _first_non_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def summarize_recstore_csv(recstore_csv: Path) -> dict[str, float]:
    rows = _load_rows(recstore_csv)
    if not rows:
        raise ValueError(f"no rows found in recstore csv: {recstore_csv}")

    emb_stage_ms = _mean(rows, "emb_stage_ms")
    dense_fwd_ms = _mean(rows, "dense_fwd_ms")
    backward_ms = _mean(rows, "backward_ms")
    optimizer_ms = _mean(rows, "optimizer_ms")
    sparse_update_ms = _mean(rows, "sparse_update_ms")
    step_total_ms = _mean(rows, "step_total_ms")
    if (
        emb_stage_ms is not None
        and dense_fwd_ms is not None
        and backward_ms is not None
        and optimizer_ms is not None
        and sparse_update_ms is not None
        and step_total_ms is not None
    ):
        return {
            "emb_stage_ms": emb_stage_ms,
            "dense_fwd_ms": dense_fwd_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
            "sparse_update_ms": sparse_update_ms,
            "step_total_ms": step_total_ms,
        }

    network_us = _first_non_none(
        _mean(rows, "network_transport_us"),
        _mean(rows, "network_framework_us_approx"),
    )
    if network_us is None:
        client_rpc_us = _mean(rows, "client_rpc_us")
        server_total_us = _mean(rows, "server_total_us")
        if client_rpc_us is not None and server_total_us is not None:
            network_us = max(0.0, client_rpc_us - server_total_us)

    kv_backend_us = _first_non_none(
        _mean(rows, "storage_backend_update_us"),
        _mean(rows, "server_backend_update_us"),
    )
    server_total_us = _mean(rows, "server_total_us")

    if network_us is None or kv_backend_us is None or server_total_us is None:
        raise ValueError(
            "recstore csv misses required columns for comparison: "
            "network_transport_us/network_framework_us_approx/client_rpc_us+server_total_us, "
            "storage_backend_update_us/server_backend_update_us, server_total_us"
        )

    return {
        "network_proxy_ms": network_us / 1000.0,
        "kv_backend_ms": kv_backend_us / 1000.0,
        "server_total_ms": server_total_us / 1000.0,
    }


def summarize_torchrec_main_csv(torchrec_main_csv: Path) -> dict[str, float]:
    rows = _load_rows(torchrec_main_csv)
    if not rows:
        raise ValueError(f"no rows found in torchrec main csv: {torchrec_main_csv}")

    emb_stage_ms = _first_non_none(
        _mean(rows, "emb_stage_ms"),
        _mean(rows, "kv_extended_ms"),
    )
    dense_fwd_ms = _mean(rows, "dense_fwd_ms")
    backward_ms = _mean(rows, "backward_ms")
    optimizer_ms = _mean(rows, "optimizer_ms")
    sparse_update_ms = _mean(rows, "sparse_update_ms")
    step_total_ms = _mean(rows, "step_total_ms")
    if (
        emb_stage_ms is not None
        and dense_fwd_ms is not None
        and backward_ms is not None
        and optimizer_ms is not None
        and sparse_update_ms is not None
        and step_total_ms is not None
    ):
        return {
            "emb_stage_ms": emb_stage_ms,
            "dense_fwd_ms": dense_fwd_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
            "sparse_update_ms": sparse_update_ms,
            "step_total_ms": step_total_ms,
        }

    network_main_ms = _mean(rows, "collective_total_ms")
    network_extended_ms = _first_non_none(
        _mean(rows, "network_proxy_torchrec_extended_ms"),
    )
    kv_local_ms = _mean(rows, "kv_local_only_ms")
    kv_extended_ms = _mean(rows, "kv_extended_ms")

    if network_main_ms is None or kv_local_ms is None or kv_extended_ms is None:
        raise ValueError(
            "torchrec main csv misses required columns: collective_total_ms, kv_local_only_ms, kv_extended_ms"
        )

    if network_extended_ms is None:
        input_pack_ms = _mean(rows, "input_pack_ms") or 0.0
        output_unpack_ms = _mean(rows, "output_unpack_ms") or 0.0
        network_extended_ms = network_main_ms + input_pack_ms + output_unpack_ms

    return {
        "network_proxy_ms": network_main_ms,
        "network_proxy_extended_ms": network_extended_ms,
        "kv_local_only_ms": kv_local_ms,
        "kv_extended_ms": kv_extended_ms,
    }


def build_compare_rows(recstore_csv: Path, torchrec_main_csv: Path) -> list[dict[str, str | float]]:
    recstore = summarize_recstore_csv(recstore_csv)
    torchrec = summarize_torchrec_main_csv(torchrec_main_csv)

    aligned_stage_keys = [
        "emb_stage_ms",
        "dense_fwd_ms",
        "backward_ms",
        "optimizer_ms",
        "sparse_update_ms",
        "step_total_ms",
    ]
    if all(key in recstore for key in aligned_stage_keys) and all(
        key in torchrec for key in aligned_stage_keys
    ):
        pairs = [
            ("emb_stage", recstore["emb_stage_ms"], torchrec["emb_stage_ms"]),
            ("dense_fwd", recstore["dense_fwd_ms"], torchrec["dense_fwd_ms"]),
            ("backward", recstore["backward_ms"], torchrec["backward_ms"]),
            ("optimizer", recstore["optimizer_ms"], torchrec["optimizer_ms"]),
            (
                "sparse_update",
                recstore["sparse_update_ms"],
                torchrec["sparse_update_ms"],
            ),
            ("step_total", recstore["step_total_ms"], torchrec["step_total_ms"]),
        ]
    else:
        pairs = [
            (
                "network_main",
                recstore["network_proxy_ms"],
                torchrec["network_proxy_ms"],
            ),
            (
                "network_extended",
                recstore["network_proxy_ms"],
                torchrec["network_proxy_extended_ms"],
            ),
            (
                "kv_strict",
                recstore["kv_backend_ms"],
                torchrec["kv_local_only_ms"],
            ),
            (
                "server_vs_extended",
                recstore["server_total_ms"],
                torchrec["kv_extended_ms"],
            ),
        ]

    rows: list[dict[str, str | float]] = []
    for metric, recstore_ms, torchrec_ms in pairs:
        delta_ms = recstore_ms - torchrec_ms
        delta_ratio = ""
        if torchrec_ms > 0:
            delta_ratio = delta_ms / torchrec_ms
        rows.append(
            {
                "metric": metric,
                "recstore_ms": recstore_ms,
                "torchrec_ms": torchrec_ms,
                "delta_ms": delta_ms,
                "delta_ratio": delta_ratio,
            }
        )

    return rows


def write_compare_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "recstore_ms", "torchrec_ms", "delta_ms", "delta_ratio"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
