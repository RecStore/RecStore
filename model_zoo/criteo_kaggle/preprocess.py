#!/usr/bin/env python3
"""Stream Criteo Kaggle train.txt into npy files matching torchrec BinaryCriteoUtils."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

INT_FEATURES = 13
CAT_FEATURES = 26
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "train.txt")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "processed")
    os.makedirs(out_dir, exist_ok=True)

    n = int(subprocess.check_output(["wc", "-l", in_path]).split()[0])
    print(f"rows={n}", flush=True)

    dense_path = os.path.join(out_dir, "train_dense.npy")
    sparse_path = os.path.join(out_dir, "train_sparse.npy")
    labels_path = os.path.join(out_dir, "train_labels.npy")

    dense = np.lib.format.open_memmap(
        dense_path, mode="w+", dtype=np.float32, shape=(n, INT_FEATURES)
    )
    sparse = np.lib.format.open_memmap(
        sparse_path, mode="w+", dtype=np.int64, shape=(n, CAT_FEATURES)
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int32, shape=(n, 1)
    )

    with open(in_path, "r", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 1 + INT_FEATURES + CAT_FEATURES:
                parts.extend([""] * (1 + INT_FEATURES + CAT_FEATURES - len(parts)))
            labels[i, 0] = int(parts[0] or "0")
            dense[i] = [int(v or "0") for v in parts[1:14]]
            sparse[i] = [int(v or "0", 16) for v in parts[14:40]]
            if i % 2_000_000 == 0:
                print(f"parsed {i}/{n}", flush=True)

    np.log(dense + 3.0, out=dense)
    n_inf = int(np.isneginf(dense).sum())
    if n_inf:
        dense[np.isneginf(dense)] = np.log(3.0)
        print(f"patched {n_inf} -inf dense cells to log(3)", flush=True)
    dense.flush()
    sparse.flush()
    labels.flush()

    for kind in ("dense", "sparse", "labels"):
        src = f"train_{kind}.npy"
        dst = os.path.join(out_dir, f"day_0_{kind}.npy")
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print("done", flush=True)


if __name__ == "__main__":
    main()
