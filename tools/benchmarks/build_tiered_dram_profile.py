#!/usr/bin/env python3
"""Build an immutable DRAM key profile from an earlier sparse-data window."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def build_profile(sparse, cards, profile_end, dram_rows, strategy,
                  key_encoding, fusion_k=30, tensor_tag=0):
    cards = np.asarray(cards, dtype=np.int64)
    if cards.ndim != 1 or not len(cards) or np.any(cards <= 0):
        raise ValueError("cardinalities must be a nonempty list of positive integers")
    if (sparse.ndim != 2 or sparse.shape[1] != len(cards)
            or not np.issubdtype(sparse.dtype, np.signedinteger)):
        raise ValueError("sparse data must be a signed-integer [samples, features] array")
    if not 0 < profile_end <= len(sparse):
        raise ValueError("profile_end must select a nonempty historical prefix")
    rows = int(cards.sum())
    if not 0 <= dram_rows <= rows:
        raise ValueError("dram_rows must be within the total row count")
    prefix = np.asarray(sparse[:profile_end], dtype=np.int64)
    if np.any(prefix == np.iinfo(np.int64).min):
        raise ValueError("INT64_MIN cannot be normalized with absolute value")
    offsets = np.r_[0, np.cumsum(cards)[:-1]]
    keys = np.abs(prefix) % cards + offsets
    counts = np.bincount(keys.ravel(), minlength=rows)
    scores = counts.astype(np.float64)
    if strategy == 'density':
        for offset, card in zip(offsets, cards):
            table = counts[offset:offset + card]
            unseen = table == 0
            estimate = np.count_nonzero(table == 1) / np.count_nonzero(unseen) if unseen.any() else 0
            scores[offset:offset + card][unseen] = estimate
    elif strategy != 'frequency':
        raise ValueError("unknown profile strategy")
    # Equal-width rows: rank by score, then deterministically by global ID.
    selected = np.sort(np.lexsort((np.arange(rows), -scores))[:dram_rows])
    if key_encoding == 'fused':
        if not 0 < fusion_k < 56 or not 0 <= tensor_tag < 256:
            raise ValueError("invalid PS fusion shift or tensor tag")
        if np.any(cards > (1 << fusion_k)) or (len(cards) - 1) << fusion_k >= (1 << 56):
            raise ValueError("fused feature IDs overflow the 56-bit row namespace")
        table_ids = np.searchsorted(offsets, selected, side='right') - 1
        selected = ((np.uint64(tensor_tag) << np.uint64(56))
                    | (table_ids.astype(np.uint64) << np.uint64(fusion_k))
                    | (selected - offsets[table_ids]).astype(np.uint64))
    elif key_encoding != 'contiguous':
        raise ValueError("unknown key encoding")
    return np.sort(selected).astype(np.uint64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sparse-path', type=Path, required=True)
    parser.add_argument('--cardinalities', type=json.loads, required=True,
                        help='JSON integer list, one cardinality per feature')
    parser.add_argument('--profile-end', type=int, required=True)
    parser.add_argument('--dram-rows', type=int, required=True)
    parser.add_argument('--strategy', choices=['frequency', 'density'], default='density')
    parser.add_argument('--key-encoding', choices=['contiguous', 'fused'], required=True)
    parser.add_argument('--fusion-k', type=int, default=30)
    parser.add_argument('--tensor-tag', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not isinstance(args.cardinalities, list) or any(type(c) is not int for c in args.cardinalities):
        parser.error('--cardinalities must be a JSON list of integers')
    sparse = np.load(args.sparse_path, mmap_mode='r', allow_pickle=False)
    selected = build_profile(sparse, args.cardinalities, args.profile_end, args.dram_rows,
                             args.strategy, args.key_encoding, args.fusion_k, args.tensor_tag)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / 'dram_keys.bin'
    selected.tofile(path)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update({'native_byteorder': sys.byteorder, 'profile_bytes': path.stat().st_size,
                     'profile_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'profile_samples': [0, args.profile_end],
                     'unseen_score': 'singleton_count / unseen_row_count per table' if args.strategy == 'density' else 0,
                     'tie_break': 'ascending contiguous global ID'})
    (args.output_dir / 'profile.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
