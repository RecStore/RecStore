from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SPARSE_FEATURES_PER_SAMPLE = 26


def _has_rdma() -> bool:
    infiniband = Path("/dev/infiniband")
    return infiniband.exists() and any(infiniband.glob("uverbs*"))


def _dense_arch_for_embedding_dim(embedding_dim: int) -> str:
    if int(embedding_dim) >= 128:
        return "512,256,128"
    return f"512,256,{int(embedding_dim)}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in _read_csv(path)]
