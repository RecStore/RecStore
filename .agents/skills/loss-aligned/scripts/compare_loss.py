#!/usr/bin/env python3
"""Compare RecStore and TorchRec loss values by rank and step."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


Key = tuple[int, int]


def load_losses(path: Path) -> dict[Key, float]:
    losses: dict[Key, float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                key = (int(row.get("rank", "0")), int(row["step"]))
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid rank, step, or loss") from exc
            if key in losses:
                raise ValueError(f"{path}:{line}: duplicate rank/step {key}")
            if not math.isfinite(loss):
                raise ValueError(f"{path}:{line}: non-finite loss for {key}")
            losses[key] = loss
    if not losses:
        raise ValueError(f"{path}: no loss rows")
    return losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recstore_csv", type=Path)
    parser.add_argument("torchrec_csv", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")

    try:
        recstore = load_losses(args.recstore_csv)
        torchrec = load_losses(args.torchrec_csv)
    except (OSError, ValueError) as exc:
        print(f"not aligned: {exc}")
        return 2

    recstore_keys = set(recstore)
    torchrec_keys = set(torchrec)
    if recstore_keys != torchrec_keys:
        print(
            "not aligned: rank/step keys differ "
            f"recstore_only={sorted(recstore_keys - torchrec_keys)[:10]} "
            f"torchrec_only={sorted(torchrec_keys - recstore_keys)[:10]}"
        )
        return 2

    mismatches: list[tuple[Key, float, float, float, float]] = []
    max_abs = 0.0
    max_rel = 0.0
    for key in sorted(recstore):
        lhs = recstore[key]
        rhs = torchrec[key]
        abs_diff = abs(lhs - rhs)
        rel_diff = abs_diff / max(abs(lhs), abs(rhs)) if lhs or rhs else 0.0
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
        if not math.isclose(lhs, rhs, abs_tol=args.atol, rel_tol=args.rtol):
            mismatches.append((key, lhs, rhs, abs_diff, rel_diff))

    print(
        f"compared={len(recstore)} atol={args.atol:g} rtol={args.rtol:g} "
        f"max_abs={max_abs:.9g} max_rel={max_rel:.9g}"
    )
    for key, lhs, rhs, abs_diff, rel_diff in mismatches[:10]:
        print(
            f"mismatch rank={key[0]} step={key[1]} recstore={lhs:.9g} "
            f"torchrec={rhs:.9g} abs={abs_diff:.9g} rel={rel_diff:.9g}"
        )
    if mismatches:
        print(f"not aligned: mismatches={len(mismatches)}")
        return 1
    print("aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
