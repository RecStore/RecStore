"""QuantaRec-style embedding backend for architecture comparison.

Faithfully replicates the QuantaRec embedding architecture inside the rs_demo
harness so it can be compared against the RecStore PS architecture with IDENTICAL
model (RankMixer), data, and timing instrumentation.

QuantaRec architecture (vs RecStore PS):
  - Embeddings live in a LOCAL hash table on every worker (replicated), not on a
    central PS.  Forward lookup is a local gather -> NO network read.
  - Gradients are synchronized SPARSELY: only the rows touched in this batch are
    communicated across workers (all-gather of unique ids + all-reduce of their
    gradients), NOT a dense all-reduce of the full table.
  - Each worker applies the optimizer update locally; because gradients are
    all-reduced identically, all replicas stay consistent (grad_reduce_by=worker).

RecStore architecture (for contrast):
  - Embeddings live on a central PS.  Forward = network pull (mitigated by
    BagPipe prefetch/GPU cache).  Update = sparse writeback to PS.
"""
from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _KeyedEmbeddings:
    """Minimal KeyedTensor-compatible wrapper: supports embeddings[name] -> [B, D]."""

    def __init__(self, tensor: torch.Tensor, keys: list[str]):
        self._tensor = tensor  # [B, F, D]
        self._keys = list(keys)
        self._map = {k: i for i, k in enumerate(self._keys)}

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._tensor[:, self._map[key], :]

    def keys(self) -> list[str]:
        return list(self._keys)

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def device(self):
        return self._tensor.device


class QuantaEmbeddingBagCollection(nn.Module):
    """Local replicated embedding table + sparse gradient all-reduce."""

    def __init__(
        self,
        embedding_bag_configs: list[dict],
        kv_client: Any = None,
        initialize_tables: bool = True,
        device: torch.device | None = None,
        fuse_k: int = 30,
        enable_fusion: bool = True,
        **_unused,
    ) -> None:
        super().__init__()
        self._configs = list(embedding_bag_configs)
        self.feature_keys: list[str] = []
        self._embedding_dims: list[int] = []
        self._num_embeddings_per_table: list[int] = []
        self._feature_table_indices: dict[str, int] = {}
        for table_idx, c in enumerate(self._configs):
            for feature_name in c["feature_names"]:
                self.feature_keys.append(feature_name)
                self._feature_table_indices[feature_name] = table_idx
                self._embedding_dims.append(int(c["embedding_dim"]))
            self._num_embeddings_per_table.append(int(c["num_embeddings"]))
        self.embedding_dim = int(self._embedding_dims[0]) if self._embedding_dims else 128
        self.num_features = len(self.feature_keys)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fuse_k = int(fuse_k)

        # Local replicated embedding tables (one per table config).
        self.tables = nn.ParameterList()
        for table_idx, c in enumerate(self._configs):
            num_emb = int(c["num_embeddings"])
            dim = int(c["embedding_dim"])
            if initialize_tables:
                tbl = torch.empty(num_emb, dim, device=self.device)
                nn.init.normal_(tbl, mean=0.0, std=0.01)
            else:
                tbl = torch.zeros(num_emb, dim, device=self.device)
            self.tables.append(nn.Parameter(tbl))

        self._single_node_forward_profile: dict[str, float] = {}
        self._perf_stats: dict[str, float] = {}
        self._last_step_profile: dict[str, float] = {}
        self._last_lookup_ids: list[torch.Tensor] = []
        self._grad_reduce_mode = "allreduce_sparse"

    def forward(self, sparse_features: Any) -> _KeyedEmbeddings:
        t0 = time.perf_counter()
        prof = {}
        bs = None
        pooled_list: list[torch.Tensor] = []
        self._last_lookup_ids = []
        for table_idx, feature_name in enumerate(self.feature_keys):
            feat = sparse_features[feature_name]
            ids = feat.values().to(self.device).long()
            if bs is None:
                bs = ids.shape[0]
            tbl = self.tables[table_idx]
            emb = tbl[ids]  # [B, D] local gather
            pooled_list.append(emb)
            uniq = torch.unique(ids)
            self._last_lookup_ids.append(uniq)

        pooled = torch.stack(pooled_list, dim=1)
        prof["quanta_lookup_local_ms"] = (time.perf_counter() - t0) * 1e3
        self._single_node_forward_profile = prof
        self._perf_stats.update(prof)
        return _KeyedEmbeddings(pooled, self.feature_keys)

    def reset_perf_stats(self) -> None:
        self._perf_stats = {}
        self._single_node_forward_profile = {}

    def consume_perf_stats(self, reset: bool = True) -> dict[str, float]:
        stats = dict(self._perf_stats)
        if reset:
            self._perf_stats = {}
        return stats

    def _can_use_single_node_distributed_fast_path(self) -> bool:
        return False


class QuantaSparseSGD(torch.optim.Optimizer):
    """Sparse SGD with cross-worker gradient all-reduce (QuantaRec architecture).

    Vectorized sparse gradient sync:
      1. Collect per-table (unique_ids, grad_rows) from the local backward.
      2. Encode (table_idx, id) into a single int64 key (table in high 32 bits).
      3. all_gather the packed [key, grad...] buffers across workers (padded).
      4. Use scatter_add_ to aggregate gradients by key (vectorized, no Python loop).
      5. Write aggregated grads back and apply local SGD update.
    """

    def __init__(self, params, lr: float = 0.01, dist=None, device=None):
        super().__init__(params, {"lr": lr})
        self._lr = lr
        self._dist = dist
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_dist = (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        self._last_step_profile: dict[str, float] = {}
        self._grad_sync_ms = 0.0
        self._update_ms = 0.0

    def _sparse_grad_sync(self, module: QuantaEmbeddingBagCollection) -> None:
        if not self._use_dist:
            return
        t0 = time.perf_counter()
        world_size = self._dist.get_world_size()

        # Collect all touched (table_idx, id, grad) into flat buffers.
        # Encode key as a single int64: (table_idx << 40) | id  (ids < 2^40).
        all_keys: list[torch.Tensor] = []
        all_grads: list[torch.Tensor] = []
        for table_idx, param in enumerate(module.tables):
            if param.grad is None or table_idx >= len(module._last_lookup_ids):
                continue
            uniq = module._last_lookup_ids[table_idx]
            if uniq.numel() == 0:
                continue
            grad_rows = param.grad[uniq]  # [n, D]
            keys = uniq.to(torch.int64) + (torch.tensor(table_idx, dtype=torch.int64, device=uniq.device) << 40)
            all_keys.append(keys)
            all_grads.append(grad_rows)

        if not all_keys:
            self._grad_sync_ms = (time.perf_counter() - t0) * 1e3
            return

        cat_keys = torch.cat(all_keys)         # [n_local] int64
        cat_grads = torch.cat(all_grads, dim=0)  # [n_local, D] float32
        n_local = cat_keys.shape[0]
        D = cat_grads.shape[1]

        # Gather counts, pad to max.
        count_t = torch.tensor([n_local], device=self._device)
        counts = [torch.zeros_like(count_t) for _ in range(world_size)]
        self._dist.all_gather(counts, count_t)
        max_n = int(max(c.item() for c in counts))

        # Exchange keys as int64 and grads as float32 (separate all_gather to
        # avoid float32 precision loss on large int keys).
        pad_keys = torch.full((max_n,), -1, dtype=torch.int64, device=self._device)
        pad_keys[:n_local] = cat_keys
        pad_grads = torch.zeros(max_n, D, device=self._device, dtype=cat_grads.dtype)
        pad_grads[:n_local] = cat_grads

        gathered_keys = [torch.zeros_like(pad_keys) for _ in range(world_size)]
        gathered_grads = [torch.zeros_like(pad_grads) for _ in range(world_size)]
        self._dist.all_gather(gathered_keys, pad_keys)
        self._dist.all_gather(gathered_grads, pad_grads.contiguous())

        # Concatenate and filter sentinels.
        all_keys_cat = torch.cat(gathered_keys)
        all_grads_cat = torch.cat(gathered_grads, dim=0)
        valid_mask = all_keys_cat >= 0
        valid_keys = all_keys_cat[valid_mask]
        valid_grads = all_grads_cat[valid_mask]

        # Aggregate by key (vectorized via unique + index_add_).
        uniq_keys, inverse = torch.unique(valid_keys, return_inverse=True)
        agg_grads = torch.zeros(uniq_keys.shape[0], D, device=self._device, dtype=valid_grads.dtype)
        agg_grads.index_add_(0, inverse, valid_grads)

        # Write aggregated gradients back to the correct table rows.
        table_idx_of = (uniq_keys >> 40).to(torch.int64)
        raw_ids = (uniq_keys & ((1 << 40) - 1)).to(torch.int64)
        for table_idx, param in enumerate(module.tables):
            if param.grad is None:
                continue
            mask = table_idx_of == table_idx
            if not mask.any():
                continue
            param.grad[raw_ids[mask]] = agg_grads[mask]

        self._grad_sync_ms = (time.perf_counter() - t0) * 1e3

    def step(self, closure=None):  # type: ignore[override]
        module = getattr(self, "_quanta_module", None)
        t0 = time.perf_counter()
        if module is not None:
            self._sparse_grad_sync(module)
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group.get("lr", self._lr)
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    param.data -= lr * param.grad.data
        self._update_ms = (time.perf_counter() - t0) * 1e3
        self._last_step_profile = {
            "quanta_sparse_grad_sync_ms": self._grad_sync_ms,
            "quanta_optimizer_step_ms": self._update_ms,
        }
        return loss

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)

    def flush(self) -> None:
        pass


def make_quanta_optimizer(module: QuantaEmbeddingBagCollection, lr: float,
                          dist=None, device=None) -> QuantaSparseSGD:
    opt = QuantaSparseSGD(module.parameters(), lr=lr, dist=dist, device=device)
    opt._quanta_module = module
    return opt
