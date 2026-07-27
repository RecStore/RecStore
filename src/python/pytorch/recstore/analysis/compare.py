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


def _mean_or_zero(rows: list[dict[str, str]], column: str) -> float:
    return _mean(rows, column) or 0.0


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

    network_main_ms = _first_non_none(
        _mean(rows, "embed_transport_ms"),
        _mean(rows, "collective_total_ms"),
    )
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


def build_exposed_gap_rows(
    recstore_csv: Path,
    torchrec_main_csv: Path,
) -> list[dict[str, str | float]]:
    """Build paper-style raw/exposed latency gap rows.

    Raw time is the measured stage cost. Exposed time is the portion not hidden
    by the measured overlap window. For RecStore prefetch, dense compute is the
    overlap window. TorchRec HBM lookup has no explicit prefetch network wait in
    this CSV layer, so its exposed prefetch network time is treated as zero.
    """

    rec_rows = _load_rows(recstore_csv)
    tr_rows = _load_rows(torchrec_main_csv)
    if not rec_rows:
        raise ValueError(f"no rows found in recstore csv: {recstore_csv}")
    if not tr_rows:
        raise ValueError(f"no rows found in torchrec main csv: {torchrec_main_csv}")

    rec_step = _mean_or_zero(rec_rows, "step_total_ms")
    tr_step = _mean_or_zero(tr_rows, "step_total_ms")
    rec_emb = _first_non_none(
        _mean(rec_rows, "emb_stage_ms"),
        _mean(rec_rows, "embed_lookup_local_ms"),
    ) or 0.0
    tr_emb = _first_non_none(
        _mean(tr_rows, "emb_stage_ms"),
        _mean(tr_rows, "embed_lookup_local_ms"),
    ) or 0.0
    rec_lookup = _first_non_none(
        _mean(rec_rows, "lookup_total_ms"),
        _mean(rec_rows, "embed_lookup_local_ms"),
    ) or 0.0
    tr_lookup = _mean_or_zero(tr_rows, "embed_lookup_local_ms")
    rec_dense = _first_non_none(
        _mean(rec_rows, "dense_compute_ms"),
        sum(
            _mean_or_zero(rec_rows, key)
            for key in ("dense_fwd_ms", "backward_ms", "optimizer_ms")
        ),
    ) or 0.0
    tr_dense = sum(
        _mean_or_zero(tr_rows, key)
        for key in ("dense_fwd_ms", "backward_ms", "optimizer_ms")
    )
    rec_prefetch_wait = _first_non_none(
        _mean(rec_rows, "prefetch_network_wait_ms"),
        (
            _mean_or_zero(rec_rows, "lookup_wait_ms")
            + _mean_or_zero(rec_rows, "planned_gpu_cache_prefill_wait_ms")
        ),
    ) or 0.0
    rec_prefetch_exposed = max(0.0, rec_prefetch_wait - rec_dense)

    metrics = [
        (
            "step_total",
            rec_step,
            rec_step,
            tr_step,
            tr_step,
            "end-to-end visible training step",
        ),
        (
            "embedding_stage",
            rec_emb,
            rec_emb,
            tr_emb,
            tr_emb,
            "input pack + lookup + pool + output unpack",
        ),
        (
            "embedding_lookup",
            rec_lookup,
            rec_lookup,
            tr_lookup,
            tr_lookup,
            "RecStore lookup path versus TorchRec HBM lookup",
        ),
        (
            "prefetch_network",
            rec_prefetch_wait,
            rec_prefetch_exposed,
            0.0,
            0.0,
            "wait not hidden by dense compute window",
        ),
        (
            "gpu_cache_query",
            _mean_or_zero(rec_rows, "lookup_gpu_cache_query_ms"),
            _mean_or_zero(rec_rows, "lookup_gpu_cache_query_ms"),
            0.0,
            0.0,
            "RecStore GPU cache lookup overhead absent from TorchRec HBM lane",
        ),
        (
            "gpu_cache_fill",
            _mean_or_zero(rec_rows, "lookup_gpu_cache_fill_ms"),
            _mean_or_zero(rec_rows, "lookup_gpu_cache_fill_ms"),
            0.0,
            0.0,
            "cache miss fill overhead",
        ),
        (
            "gpu_cache_prefill",
            _mean_or_zero(rec_rows, "planned_gpu_cache_prefill_ms"),
            _mean_or_zero(rec_rows, "planned_gpu_cache_prefill_ms"),
            0.0,
            0.0,
            "lookahead result insertion into GPU cache",
        ),
        (
            "sparse_update",
            _mean_or_zero(rec_rows, "sparse_update_ms"),
            _mean_or_zero(rec_rows, "sparse_update_ms"),
            _mean_or_zero(tr_rows, "sparse_update_ms"),
            _mean_or_zero(tr_rows, "sparse_update_ms"),
            "sparse gradient replay/apply/flush path",
        ),
        (
            "gpu_cache_invalidate",
            _mean_or_zero(rec_rows, "update_gpu_cache_invalidate_ms"),
            _mean_or_zero(rec_rows, "update_gpu_cache_invalidate_ms"),
            0.0,
            0.0,
            "read-after-write cache invalidation cost",
        ),
        (
            "dense_compute",
            rec_dense,
            rec_dense,
            tr_dense,
            tr_dense,
            "dense forward + backward + dense optimizer",
        ),
    ]

    rows: list[dict[str, str | float]] = []
    for metric, rec_raw, rec_exposed, tr_raw, tr_exposed, note in metrics:
        delta_raw = rec_raw - tr_raw
        delta_exposed = rec_exposed - tr_exposed
        bottleneck = "hidden"
        if abs(delta_exposed) > 1e-12:
            bottleneck = "exposed"
        elif abs(delta_raw) > 1e-12:
            bottleneck = "raw_only"
        rows.append(
            {
                "metric": metric,
                "recstore_raw_ms": rec_raw,
                "recstore_exposed_ms": rec_exposed,
                "torchrec_raw_ms": tr_raw,
                "torchrec_exposed_ms": tr_exposed,
                "delta_raw_ms": delta_raw,
                "delta_exposed_ms": delta_exposed,
                "bottleneck": bottleneck,
                "note": note,
            }
        )
    return rows


def write_compare_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else [
            "metric",
            "recstore_ms",
            "torchrec_ms",
            "delta_ms",
            "delta_ratio",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
