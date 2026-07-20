"""Analyze RecStore (PS + bagpipe) vs TorchRec (local embeddings) RankMixer runs.

Reads the recstore_main.csv and torchrec_main.csv produced by run_mock_stress.py
for the two architectures and prints a side-by-side timing comparison focused on
the embedding-architecture difference (lookup, sparse update, end-to-end step).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Columns of interest (present in both backends' CSVs).
TIMING_COLS = [
    ("embed_lookup_local_ms", "embedding lookup"),
    ("embed_pool_local_ms", "embedding pool/reshape"),
    ("dense_fwd_ms", "dense forward (RankMixer)"),
    ("backward_ms", "backward"),
    ("optimizer_ms", "dense optimizer"),
    ("sparse_update_ms", "sparse update + writeback"),
    ("batch_prepare_ms", "batch prepare"),
]
STEP_TOTAL_HINT = "dense_fwd_ms"  # used only for sanity


def _read_non_warmup(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows = [r for r in csv.DictReader(open(csv_path))
            if r.get("warmup_excluded") == "0" and r.get("step") is not None]
    return rows


def _stats(rows: list[dict], col: str):
    vals = []
    for r in rows:
        v = r.get(col)
        if v is None or v == "":
            continue
        try:
            vals.append(float(v))
        except ValueError:
            pass
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    p50 = vals[n // 2]
    p95 = vals[min(n - 1, int(n * 0.95))]
    return {"mean": mean, "p50": p50, "p95": p95, "n": n}


def _step_total(rows: list[dict]) -> float | None:
    """Approximate end-to-end step time from the available stage timers."""
    if not rows:
        return None
    totals = []
    for r in rows:
        t = 0.0
        for col, _ in TIMING_COLS:
            v = r.get(col)
            if v and v != "":
                try:
                    t += float(v)
                except ValueError:
                    pass
        totals.append(t)
    if not totals:
        return None
    return sum(totals) / len(totals)


def _fmt(v: float | None) -> str:
    return f"{v:8.3f}" if v is not None else "     n/a"


def compare(recstore_csv: Path, torchrec_csv: Path) -> int:
    rs = _read_non_warmup(recstore_csv)
    tr = _read_non_warmup(torchrec_csv)
    if not rs:
        print(f"WARNING: no non-warmup rows in {recstore_csv}")
    if not tr:
        print(f"WARNING: no non-warmup rows in {torchrec_csv}")

    print("=" * 78)
    print("RankMixer performance: RecStore (PS + bagpipe) vs TorchRec (local emb)")
    print("=" * 78)
    print(f"  RecStore steps: {len(rs)}   TorchRec steps: {len(tr)}")
    print()
    header = f"{'stage':<32} {'RecStore mean':>14} {'TorchRec mean':>14} {'ratio (RS/TR)':>14}"
    print(header)
    print("-" * len(header))
    for col, label in TIMING_COLS:
        s_rs = _stats(rs, col)
        s_tr = _stats(tr, col)
        m_rs = s_rs["mean"] if s_rs else None
        m_tr = s_tr["mean"] if s_tr else None
        ratio = (m_rs / m_tr) if (m_rs and m_tr and m_tr > 0) else None
        ratio_s = f"{ratio:8.2f}x" if ratio else "       n/a"
        print(f"{label:<32} {_fmt(m_rs):>14} {_fmt(m_tr):>14} {ratio_s:>14}")

    print("-" * len(header))
    t_rs = _step_total(rs)
    t_tr = _step_total(tr)
    ratio = (t_rs / t_tr) if (t_rs and t_tr and t_tr > 0) else None
    ratio_s = f"{ratio:8.2f}x" if ratio else "       n/a"
    print(f"{'sum of stages (per step)':<32} {_fmt(t_rs):>14} {_fmt(t_tr):>14} {ratio_s:>14}")

    # Loss sanity (accuracy parity check).
    print()
    print("Loss (accuracy parity — should be ~equal across architectures):")
    for label, rows in [("RecStore", rs), ("TorchRec", tr)]:
        losses = []
        for r in rows:
            v = r.get("loss")
            if v and v != "":
                try:
                    losses.append(float(v))
                except ValueError:
                    pass
        if losses:
            print(f"  {label}: first={losses[0]:.4f} last={losses[-1]:.4f} "
                  f"delta={losses[-1]-losses[0]:+.4f} (n={len(losses)})")

    # Bagpipe stats (RecStore only).
    if rs:
        print()
        print("RecStore bagpipe cache stats (non-warmup mean):")
        for col in ["bagpipe_cache_entries", "bagpipe_dirty_entries",
                    "bagpipe_prefetch_batches", "bagpipe_prefetch_ids",
                    "bagpipe_prefetch_skip_cached", "bagpipe_prefetch_pruned"]:
            st = _stats(rs, col)
            if st:
                print(f"  {col:<32} mean={st['mean']:.1f}")
    print("=" * 78)
    return 0


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        return compare(Path(args[0]), Path(args[1]))
    # Default paths from a typical bench run.
    root = Path("/tmp/rs_rankmixer_bench")
    rs = root / "recstore/outputs/bench/recstore_main.csv"
    tr = root / "torchrec/outputs/bench/torchrec_main.csv"
    return compare(rs, tr)


if __name__ == "__main__":
    raise SystemExit(main())
