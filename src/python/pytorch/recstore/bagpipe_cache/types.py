"""Data structures and work-handle objects for BagPipe cache controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class CacheEntry:
    """Per-entry metadata for a fused ID currently residing in the GPU cache."""
    ttl: int            # batch number at which this entry expires
    dirty: bool = False # whether it received an SGD update (needs writeback)


@dataclass
class PrefetchSlot:
    """A batched prefetch in flight."""
    handle: int
    ids_cpu: torch.Tensor       # unique fused IDs prefetched
    ttl_map: Dict[int, int]     # fused_id -> expiry batch
    issue_ts: float
    num_ids: int


class _CompositeWork:
    """Work handle for sparse all_gather + aggregate across GPUs."""

    def __init__(self, work_ids, work_grads, ids_list, grads_list,
                 n_list, world_size, dim, device, stats):
        self._work_ids = work_ids
        self._work_grads = work_grads
        self._ids_list = ids_list
        self._grads_list = grads_list
        self._n_list = n_list
        self._world_size = world_size
        self._dim = dim
        self._device = device
        self._stats = stats
        self._result = None

    def wait(self):
        self._work_ids.wait()
        self._work_grads.wait()
        all_ids = []
        all_grads = []
        for r in range(self._world_size):
            nr = int(self._n_list[r].item())
            if nr > 0:
                all_ids.append(self._ids_list[r][:nr])
                all_grads.append(self._grads_list[r][:nr])
        if not all_ids:
            self._result = (
                torch.tensor([], dtype=torch.int64, device=self._device),
                torch.zeros(0, self._dim, device=self._device),
            )
            return
        cat_ids = torch.cat(all_ids)
        cat_grads = torch.cat(all_grads)
        unique_ids, inverse = torch.unique(cat_ids, return_inverse=True)
        aggregated = torch.zeros(
            (len(unique_ids), self._dim), dtype=cat_grads.dtype, device=cat_grads.device)
        aggregated.index_add_(0, inverse, cat_grads)
        self._result = (unique_ids, aggregated)

    @property
    def result(self):
        return self._result


class _DenseWork:
    """Work handle for dense all_reduce that extracts reduced grads for our IDs."""

    def __init__(self, work, dense_grads, valid_indices, ids, dim, device):
        self._work = work
        self._dense_grads = dense_grads
        self._valid_indices = valid_indices
        self._ids = ids
        self._dim = dim
        self._device = device
        self._result = None

    def wait(self):
        self._work.wait()
        reduced = self._dense_grads.index_select(0, self._valid_indices)
        self._result = (self._ids.to(self._device), reduced)

    @property
    def result(self):
        return self._result
