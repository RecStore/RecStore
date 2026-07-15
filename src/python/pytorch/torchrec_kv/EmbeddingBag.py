import torch
import logging
import time
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Dict, Any, Tuple
from time import perf_counter
from dataclasses import dataclass

try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
except ModuleNotFoundError:
    class KeyedJaggedTensor:  # pragma: no cover - fallback typing surface
        pass

    class KeyedTensor:
        def __init__(self, *, keys: List[str], values: torch.Tensor, length_per_key: List[int]) -> None:
            self._keys = list(keys)
            self._values = values
            self._length_per_key = list(length_per_key)
            self._offsets: Dict[str, tuple[int, int]] = {}
            offset = 0
            for key, width in zip(self._keys, self._length_per_key):
                width_int = int(width)
                self._offsets[key] = (offset, offset + width_int)
                offset += width_int

        def keys(self) -> List[str]:
            return list(self._keys)

        def values(self) -> torch.Tensor:
            return self._values

        def length_per_key(self) -> List[int]:
            return list(self._length_per_key)

        def __getitem__(self, key: str) -> torch.Tensor:
            start, end = self._offsets[key]
            return self._values[:, start:end]

    @dataclass
    class EmbeddingBagConfig:
        name: str
        embedding_dim: int
        num_embeddings: int
        feature_names: List[str]
from ..recstore.KVClient import get_kv_client, RecStoreClient
from ..recstore.single_node_exchange import (
    LookupEmbeddingResponsePayload,
    LookupIdsPayload,
    exchange_lookup_embedding_responses,
    exchange_lookup_ids,
    reassemble_lookup_embedding_responses,
)

logger = logging.getLogger(__name__)

_LOCAL_FAST_PATH_BACKENDS = {"local_shm", "hierkv"}

_EMPTY_LOOKUP_PROFILE: Dict[str, float] = {
    "lookup_exchange_ids_ms": 0.0,
    "lookup_local_lookup_ms": 0.0,
    "lookup_exchange_responses_ms": 0.0,
    "lookup_rebuild_ms": 0.0,
    "lookup_post_rebuild_h2d_ms": 0.0,
}


def _new_lookup_profile() -> Dict[str, float]:
    return dict(_EMPTY_LOOKUP_PROFILE)


def _merge_profile_values(dst: Dict[str, float], src: Dict[str, Any] | None) -> None:
    if not isinstance(src, dict):
        return
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0.0) + float(value)


class _RecStoreEBCFunction(Function):
    @staticmethod
    def forward(
        ctx,
        module: "RecStoreEmbeddingBagCollection",
        feature_keys: List[str],
        features_values: torch.Tensor,
        features_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(features_values, features_lengths)
        ctx.module = module
        ctx.feature_keys = feature_keys

        config = module.embedding_bag_configs()[0]
        
        all_embeddings = module.kv_client.pull(name=config.name, ids=features_values)

        all_embeddings.requires_grad = True

        local_indices = torch.arange(
            len(features_values), device=features_values.device, dtype=torch.long
        )

        offsets = torch.cat(
            [torch.tensor([0], device=features_lengths.device), torch.cumsum(features_lengths, 0)[:-1]]
        )

        pooled_output = F.embedding_bag(
            input=local_indices,
            weight=all_embeddings,
            offsets=offsets,
            mode="sum",
            sparse=False,
        )

        batch_size = features_lengths.numel() // len(module.feature_keys)
        embedding_dim = config.embedding_dim
        
        return pooled_output.view(batch_size, len(feature_keys), embedding_dim)

    @staticmethod
    def backward(
        ctx, grad_output_values: torch.Tensor
    ) -> Tuple[None, None, None, None]:
        features_values, features_lengths = ctx.saved_tensors
        module: "RecStoreEmbeddingBagCollection" = ctx.module
        feature_keys: List[str] = ctx.feature_keys

        batch_size = features_lengths.numel() // len(feature_keys)
        grad_output_reshaped = grad_output_values.view(batch_size, len(feature_keys), -1)

        lengths_cpu = features_lengths.cpu()
        values_cpu = features_values.cpu()
        
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths_cpu, 0)])

        for i, key in enumerate(feature_keys):
            config_name = module._config_names[key]
            
            for sample_idx in range(batch_size):
                feature_in_batch_idx = sample_idx * len(feature_keys) + i
                start, end = offsets[feature_in_batch_idx], offsets[feature_in_batch_idx + 1]
                
                num_items_in_bag = end - start
                if num_items_in_bag > 0:
                    ids_to_update = values_cpu[start:end]
                    grad_for_bag = grad_output_reshaped[sample_idx, i]
                    module._trace.append(
                        {
                            "name": config_name,
                            "ids": ids_to_update.detach(),
                            "grad": grad_for_bag.detach(),
                            "count": int(num_items_in_bag),
                        }
                    )
        return None, None, None, None


class RecStoreEmbeddingBagCollection(torch.nn.Module):
    def __init__(self, embedding_bag_configs: List[Dict[str, Any]], lr: float = 0.01,
                 enable_fusion: bool = True, fusion_k: int = 30,
                 ps_host: str = None, ps_port: int = None,
                 kv_client: RecStoreClient | None = None,
                 initialize_tables: bool = True):
        super().__init__()
        self._embedding_bag_configs = [
            EmbeddingBagConfig(**c) for c in embedding_bag_configs
        ]
        self.kv_client: RecStoreClient = kv_client if kv_client is not None else get_kv_client()
        if ps_host is not None and ps_port is not None:
            self.kv_client.set_ps_config(ps_host, ps_port)

        self._lr = lr
        self._enable_fusion = enable_fusion
        self._fusion_k = fusion_k
        
        self.feature_keys: List[str] = []
        self._config_names: Dict[str, str] = {}
        self._config_by_feature: Dict[str, EmbeddingBagConfig] = {}
        self._feature_table_indices: Dict[str, int] = {}
        self._embedding_dims: List[int] = [] 
        for table_idx, c in enumerate(self._embedding_bag_configs):
            for feature_name in c.feature_names:
                self.feature_keys.append(feature_name)
                self._config_names[feature_name] = c.name
                self._config_by_feature[feature_name] = c
                self._feature_table_indices[feature_name] = table_idx
                self._embedding_dims.append(c.embedding_dim)

        # Use the first table's config as the "master" to obtain embedding_dim
        # for fused reads/writes (backend only needs dim; name is not used).
        self._master_config = self._embedding_bag_configs[0] if len(self._embedding_bag_configs) > 0 else None

        self._trace = []
        self._prefetch_handles: Dict[str, int] = {}
        # Fused prefetch handle (single handle for all fused IDs)
        self._fused_prefetch_handle: int | None = None
        self._fused_prefetch_num_ids: int = 0
        self._fused_prefetch_issue_ts: float | None = None
        self._fused_prefetch_slots: List[Dict[str, Any]] = []
        # Prefetch performance stats
        self._prefetch_issue_ts: Dict[int, float] = {}  # handle -> issue time
        self._prefetch_sizes: Dict[int, int] = {}       # handle -> number of ids
        self._prefetch_wait_latencies: List[float] = []
        self._prefetch_issue_latencies: List[float] = []  # currently producer-side if provided
        self._prefetch_total_ids: int = 0

        # Cache for fused prefetch metadata (unique IDs and inverse) for one batch
        self._fused_ids_cpu: torch.Tensor | None = None
        self._fused_inverse: torch.Tensor | None = None

        # Phase-1 single-node distributed fast path stays fully opt-in.
        self.enable_single_node_distributed_fast_path: bool = False
        self.single_node_distributed_mode: str | None = None
        self.single_node_owner_policy: str = "hash_mod_world_size"
        self.single_node_ps_backend: str = "local_shm"
        self.reset_perf_stats()

        for idx, config in enumerate(self._embedding_bag_configs):
            base_offset = (idx << self._fusion_k) if self._enable_fusion else 0
            if initialize_tables:
                self.kv_client.init_data(
                    name=config.name,
                    shape=(config.num_embeddings, config.embedding_dim),
                    dtype=torch.float32,
                    base_offset=base_offset,
                )
                continue

            register_tensor_meta = getattr(self.kv_client, "register_tensor_meta", None)
            if register_tensor_meta is None:
                raise RuntimeError(
                    "initialize_tables=False requires kv_client.register_tensor_meta support"
                )
            register_tensor_meta(
                name=config.name,
                shape=(config.num_embeddings, config.embedding_dim),
                dtype=torch.float32,
                base_offset=base_offset,
            )
            # RDMA backend: the client<->server transport is established through a
            # collective QP-metadata rendezvous that the server runs per lane and
            # per client. Non-initializer ranks (which otherwise only record local
            # tensor metadata) must also open their connection here, otherwise the
            # server's rendezvous blocks forever waiting for the missing client and
            # the job deadlocks against the trainer's NCCL collectives. Calling
            # init_embedding_table only opens the connection and (idempotently)
            # registers the table; it writes no embedding data, so the
            # rank-0-only data initialization above is preserved.
            if self._remote_backend_requires_eager_connect():
                init_embedding_table = getattr(
                    self.kv_client, "init_embedding_table", None
                )
                if callable(init_embedding_table):
                    init_embedding_table(
                        config.name,
                        int(config.num_embeddings),
                        int(config.embedding_dim),
                    )

    def _remote_backend_requires_eager_connect(self) -> bool:
        """Whether non-initializer ranks must eagerly open the PS connection.

        Only the RDMA backend needs this: its transport setup is a collective
        rendezvous across all clients, so every rank must connect during table
        setup. The single-node fast path and request-lazy backends (BRPC/GRPC)
        do not, and are left untouched.
        """
        if getattr(self, "enable_single_node_distributed_fast_path", False):
            return False
        current_ps_backend = getattr(self.kv_client, "current_ps_backend", None)
        if not callable(current_ps_backend):
            return False
        try:
            backend = str(current_ps_backend()).lower()
        except Exception:
            return False
        return "rdma" in backend

    def embedding_bag_configs(self):
        return self._embedding_bag_configs

    def reset_trace(self):
        self._trace = []

    def reset_perf_stats(self) -> None:
        self._perf_stats: Dict[str, float] = {
            "prefetch_issue_ms": 0.0,
            "planned_gpu_cache_prefill_ms": 0.0,
            "planned_gpu_cache_prefill_wait_ms": 0.0,
            "planned_gpu_cache_prefill_batches": 0.0,
            "planned_gpu_cache_prefill_ids": 0.0,
            "planned_gpu_cache_prefill_successes": 0.0,
            "planned_gpu_cache_prefill_fallbacks": 0.0,
            "planned_gpu_cache_prefill_no_cuda": 0.0,
            "planned_gpu_cache_prefill_no_api": 0.0,
            "planned_gpu_cache_prefill_wait_failures": 0.0,
            "planned_gpu_cache_prefill_result_size_mismatches": 0.0,
            "lookup_ids_build_ms": 0.0,
            "lookup_wait_ms": 0.0,
            "lookup_total_ms": 0.0,
            "lookup_owner_exchange_ms": 0.0,
            "lookup_local_lookup_ms": 0.0,
            "lookup_reassemble_ms": 0.0,
            "lookup_gpu_cache_pull_ms": 0.0,
            "lookup_fallback_pull_ms": 0.0,
            "lookup_partial_merge_ms": 0.0,
            "lookup_partial_prefetched_ids": 0.0,
            "lookup_partial_missing_ids": 0.0,
            "pool_embedding_bag_ms": 0.0,
        }

    def _perf_add(self, key: str, delta_ms: float) -> None:
        self._perf_stats[key] = self._perf_stats.get(key, 0.0) + float(delta_ms)

    def consume_perf_stats(self, reset: bool = True) -> Dict[str, float]:
        stats = dict(self._perf_stats)
        if reset:
            self.reset_perf_stats()
        return stats

    def _append_trace(self, name: str, ids: torch.Tensor, grad: torch.Tensor) -> None:
        grad_view = grad.detach().to(torch.float32)
        ids_view = ids.detach().to(device=grad_view.device, dtype=torch.int64)
        if ids_view.numel() == 0:
            return
        self._trace.append(
            {
                "name": name,
                "ids": ids_view,
                "grads": grad_view,
            }
        )

    def _config_for_feature(self, key: str) -> EmbeddingBagConfig:
        return self._config_by_feature[key]

    def _dedupe_fused_ids_cpu(
        self, fused_ids_cpu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if fused_ids_cpu.numel() > 0:
            return torch.unique(fused_ids_cpu, return_inverse=True)
        return fused_ids_cpu, fused_ids_cpu

    def _gpu_cache_prefill_api(self):
        prefill = getattr(self.kv_client, "prefill_gpu_cache", None)
        is_gpu_cache_enabled = getattr(self.kv_client, "is_gpu_cache_enabled", None)
        if callable(is_gpu_cache_enabled) and not bool(is_gpu_cache_enabled()):
            self._perf_add("planned_gpu_cache_prefill_no_api", 1.0)
            self._perf_add("planned_gpu_cache_prefill_fallbacks", 1.0)
            return None
        if not callable(prefill):
            self._perf_add("planned_gpu_cache_prefill_no_api", 1.0)
            self._perf_add("planned_gpu_cache_prefill_fallbacks", 1.0)
            return None
        return prefill

    def _consume_feature_prefetch_handle(
        self, key: str, values: torch.Tensor, *, record_lookup_wait: bool = False
    ) -> torch.Tensor:
        handle = self._prefetch_handles.pop(key)
        config = self._config_for_feature(key)
        t_wait_start = perf_counter()
        emb = self.kv_client.wait_and_get(
            handle, config.embedding_dim, device=values.device
        )
        t_wait_end = perf_counter()
        if handle in self._prefetch_issue_ts:
            wait_latency = t_wait_end - t_wait_start
            issue_latency = t_wait_start - self._prefetch_issue_ts.get(
                handle, t_wait_start
            )
            self._prefetch_wait_latencies.append(wait_latency)
            self._prefetch_issue_latencies.append(issue_latency)
        if record_lookup_wait:
            self._perf_add("lookup_wait_ms", (t_wait_end - t_wait_start) * 1e3)
        return emb

    def _clear_per_feature_prefetch_state(self) -> None:
        self._prefetch_handles.clear()

    def _clear_fused_prefetch_state(self) -> None:
        self._fused_prefetch_slots.clear()
        self._sync_fused_prefetch_slot_state()

    def _clear_prefetch_state(self) -> None:
        self._clear_per_feature_prefetch_state()
        self._clear_fused_prefetch_state()

    def _pop_current_fused_prefetch_slot(self) -> None:
        if self._fused_prefetch_slots:
            self._fused_prefetch_slots.pop(0)
        self._sync_fused_prefetch_slot_state()

    def set_prefetch_handles(self, handles: Dict[str, Any]):
        """Set prefetch handles plus optional stats metadata.

        Accepts: { feature_key: handle } OR { feature_key: (handle, num_ids, issue_ts) }
        """
        if not handles:
            self._clear_prefetch_state()
            return
        self._clear_fused_prefetch_state()
        parsed: Dict[str, int] = {}
        now = time.time()
        for k, v in handles.items():
            if isinstance(v, tuple):
                if len(v) == 3:
                    h, num_ids, t_issue = v
                elif len(v) == 2:
                    h, num_ids = v; t_issue = now
                else:
                    h = v[0]; num_ids = 0; t_issue = now
                parsed[k] = int(h)
                self._prefetch_issue_ts[int(h)] = float(t_issue)
                self._prefetch_sizes[int(h)] = int(num_ids)
                self._prefetch_total_ids += int(num_ids)
            else:
                parsed[k] = int(v)
                # Unknown size, leave stats partial
        self._prefetch_handles = parsed

    def set_fused_prefetch_handle(
        self,
        handle: int,
        num_ids: int | None = None,
        issue_ts: float | None = None,
        *,
        record_stats: bool = True,
        fused_ids_cpu: torch.Tensor | None = None,
        fused_inverse: torch.Tensor | None = None,
        invalid_fused_ids_cpu: torch.Tensor | None = None,
    ):
        """Set a single fused prefetch handle for the upcoming forward.

        Optionally record stats: number of ids and issue timestamp for latency accounting.
        Extra fused_ids_cpu and fused_inverse are accepted for API compatibility with the
        prefetcher; they are currently unused but kept to avoid argument errors when
        passed through from the producer thread.
        """
        self._clear_per_feature_prefetch_state()
        slot = {
            "handle": int(handle),
            "num_ids": int(num_ids) if num_ids is not None else 0,
            "issue_ts": float(issue_ts) if issue_ts is not None else time.time(),
            "fused_ids_cpu": fused_ids_cpu if fused_ids_cpu is not None else None,
            "fused_inverse": fused_inverse if fused_inverse is not None else None,
            "invalid_fused_ids_cpu": (
                invalid_fused_ids_cpu.detach().to(dtype=torch.int64, device="cpu").flatten()
                if isinstance(invalid_fused_ids_cpu, torch.Tensor)
                else None
            ),
        }
        self._fused_prefetch_slots.append(slot)
        self._sync_fused_prefetch_slot_state()
        if record_stats:
            # Track in shared stats maps for unified reporting
            self._prefetch_issue_ts[slot["handle"]] = slot["issue_ts"]
            if slot["num_ids"]:
                self._prefetch_sizes[slot["handle"]] = slot["num_ids"]
                self._prefetch_total_ids += slot["num_ids"]

    def _sync_fused_prefetch_slot_state(self) -> None:
        if self._fused_prefetch_slots:
            slot = self._fused_prefetch_slots[0]
            self._fused_prefetch_handle = slot["handle"]
            self._fused_prefetch_num_ids = slot["num_ids"]
            self._fused_prefetch_issue_ts = slot["issue_ts"]
            self._fused_ids_cpu = slot["fused_ids_cpu"]
            self._fused_inverse = slot["fused_inverse"]
            self._fused_invalid_ids_cpu = slot.get("invalid_fused_ids_cpu")
        else:
            self._fused_prefetch_handle = None
            self._fused_prefetch_num_ids = 0
            self._fused_prefetch_issue_ts = None
            self._fused_ids_cpu = None
            self._fused_inverse = None
            self._fused_invalid_ids_cpu = None

    def _issue_fused_prefetch_from_ids(
        self,
        unique_ids: torch.Tensor,
        inverse: torch.Tensor,
        num_ids: int,
        *,
        record_handle: bool,
    ) -> int | Tuple[int, int, float, torch.Tensor, torch.Tensor]:
        if unique_ids.numel() == 0:
            issue_ts = perf_counter()
            if record_handle:
                self.set_prefetch_handles({})
                return 0
            return 0, 0, issue_ts, unique_ids, inverse
        t_issue_start = perf_counter()
        set_prefetch_table_name = getattr(self.kv_client, "set_prefetch_table_name", None)
        if callable(set_prefetch_table_name):
            set_prefetch_table_name(self._master_config.name)
        handle = self.kv_client.prefetch(unique_ids)
        self._perf_add("prefetch_issue_ms", (perf_counter() - t_issue_start) * 1e3)
        issue_ts = perf_counter()
        if record_handle:
            self.set_fused_prefetch_handle(
                handle,
                num_ids=num_ids,
                issue_ts=issue_ts,
                fused_ids_cpu=unique_ids,
                fused_inverse=inverse,
            )
            return handle
        return handle, num_ids, issue_ts, unique_ids, inverse

    def issue_fused_prefetch(
        self,
        features: KeyedJaggedTensor,
        *,
        record_handle: bool = True,
    ) -> int | Tuple[int, int, float, torch.Tensor, torch.Tensor]:
        """Compute fused global IDs and issue a single prefetch.

        When record_handle is True (default), the handle is stored on the module and the
        handle is returned. When False, the caller receives a tuple with metadata so the
        consumer can set the handle later without touching shared state in the producer
        thread: (handle, num_ids, issue_ts, fused_ids_cpu, inverse).
        """
        if not self._enable_fusion or self._master_config is None:
            raise RuntimeError("Fused prefetch requires fusion enabled and a valid master config.")

        t_build_start = perf_counter()
        keys_in_batch = list(features.keys())
        fused_values_list: List[torch.Tensor] = []
        device = features.device()
        for key in keys_in_batch:
            kjt_per_feature = features[key]
            values = kjt_per_feature.values()
            if values.dtype != torch.int64:
                values = values.to(torch.int64)
            if values.numel() > 0:
                table_idx = self._feature_table_indices[key]
                prefix = (table_idx << self._fusion_k)
                fused_values = values + prefix
                fused_values_list.append(fused_values)
        fused_values_all = torch.cat(fused_values_list, dim=0) if len(fused_values_list) > 0 else torch.empty((0,), dtype=torch.int64, device=device)
        fused_ids_cpu_full = fused_values_all.to("cpu") if fused_values_all.numel() > 0 else fused_values_all
        unique_ids, inverse = self._dedupe_fused_ids_cpu(fused_ids_cpu_full)
        self._perf_add("lookup_ids_build_ms", (perf_counter() - t_build_start) * 1e3)
        return self._issue_fused_prefetch_from_ids(
            unique_ids,
            inverse,
            int(fused_values_all.numel()),
            record_handle=record_handle,
        )

    def issue_fused_id_prefetch(
        self,
        fused_ids: torch.Tensor,
        *,
        record_handle: bool = True,
    ) -> int | Tuple[int, int, float, torch.Tensor, torch.Tensor]:
        """Issue a fused prefetch from an oracle-selected fused-id tensor."""
        if not self._enable_fusion or self._master_config is None:
            raise RuntimeError("Fused prefetch requires fusion enabled and a valid master config.")
        if not isinstance(fused_ids, torch.Tensor):
            raise TypeError("fused_ids must be a torch.Tensor")

        t_build_start = perf_counter()
        fused_ids_cpu_full = fused_ids.detach().to(dtype=torch.int64, device="cpu").flatten()
        unique_ids, inverse = self._dedupe_fused_ids_cpu(fused_ids_cpu_full)
        self._perf_add("lookup_ids_build_ms", (perf_counter() - t_build_start) * 1e3)
        return self._issue_fused_prefetch_from_ids(
            unique_ids,
            inverse,
            int(unique_ids.numel()),
            record_handle=record_handle,
        )

    def prefill_gpu_cache_for_fused_ids(self, fused_ids: torch.Tensor) -> bool:
        """Synchronously insert oracle-selected fused ids into the GPU cache."""
        if not self._enable_fusion or self._master_config is None:
            raise RuntimeError("GPU cache prefill requires fusion enabled and a valid master config.")
        if not isinstance(fused_ids, torch.Tensor):
            raise TypeError("fused_ids must be a torch.Tensor")
        if fused_ids.numel() == 0:
            return True
        prefill = self._gpu_cache_prefill_api()
        if prefill is None:
            return False

        t_prefill_start = perf_counter()
        unique_ids = torch.unique(
            fused_ids.detach().to(dtype=torch.int64, device="cpu").flatten(),
            sorted=True,
        )
        values = self.kv_client.pull(name=self._master_config.name, ids=unique_ids)
        if torch.cuda.is_available():
            cache_device = torch.device("cuda", torch.cuda.current_device())
            prefill_ids = unique_ids.to(cache_device)
            values = values.to(cache_device)
        else:
            prefill_ids = unique_ids
        if not prefill_ids.is_contiguous():
            prefill_ids = prefill_ids.contiguous()
        if not values.is_contiguous():
            values = values.contiguous()
        prefill(self._master_config.name, prefill_ids, values)
        self._perf_add("planned_gpu_cache_prefill_ms", (perf_counter() - t_prefill_start) * 1e3)
        self._perf_add("planned_gpu_cache_prefill_batches", 1.0)
        self._perf_add("planned_gpu_cache_prefill_ids", float(unique_ids.numel()))
        self._perf_add("planned_gpu_cache_prefill_successes", 1.0)
        return True

    def invalidate_gpu_cache_for_fused_ids(self, fused_ids: torch.Tensor) -> bool:
        """Invalidate oracle-evicted fused ids from the GPU cache."""
        if not self._enable_fusion or self._master_config is None:
            raise RuntimeError("GPU cache invalidation requires fusion enabled and a valid master config.")
        if not isinstance(fused_ids, torch.Tensor):
            raise TypeError("fused_ids must be a torch.Tensor")
        if fused_ids.numel() == 0:
            return True
        invalidator = getattr(self.kv_client, "invalidate_gpu_cache", None)
        is_gpu_cache_enabled = getattr(self.kv_client, "is_gpu_cache_enabled", None)
        if callable(is_gpu_cache_enabled) and not bool(is_gpu_cache_enabled()):
            return False
        if not callable(invalidator):
            return False
        unique_ids = torch.unique(
            fused_ids.detach().to(dtype=torch.int64, device="cpu").flatten(),
            sorted=True,
        )
        if torch.cuda.is_available():
            unique_ids = unique_ids.to(torch.device("cuda", torch.cuda.current_device()))
        invalidator(self._master_config.name, unique_ids)
        return True

    def _can_use_single_node_distributed_fast_path(self) -> bool:
        if not self.enable_single_node_distributed_fast_path:
            return False
        dist = getattr(torch, "distributed", None)
        if dist is None or not hasattr(dist, "is_initialized") or not dist.is_initialized():
            return False
        if not hasattr(dist, "get_world_size") or dist.get_world_size() <= 1:
            return False
        if self.single_node_distributed_mode != "single_node":
            return False
        if self.single_node_owner_policy != "hash_mod_world_size":
            return False
        if self.single_node_ps_backend not in _LOCAL_FAST_PATH_BACKENDS:
            return False
        return True

    def _uses_shared_local_shm_single_table(self) -> bool:
        probe = getattr(self.kv_client, "is_shared_local_shm_table", None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            return False

    def _prepare_single_node_local_shm_fast_path_client(self, rank: int = 0) -> None:
        current_backend = None
        if hasattr(self.kv_client, "current_ps_backend"):
            current_backend = self.kv_client.current_ps_backend()
        if hasattr(self.kv_client, "activate_shard"):
            self.kv_client.activate_shard(rank)
        if current_backend != self.single_node_ps_backend:
            if hasattr(self.kv_client, "set_ps_backend"):
                self.kv_client.set_ps_backend(self.single_node_ps_backend)

    def _build_lookup_request_payload(
        self,
        fused_ids: torch.Tensor,
        *,
        rank: int,
        world_size: int,
    ) -> LookupIdsPayload:
        normalized_ids = fused_ids.to(dtype=torch.int64)
        if not normalized_ids.is_contiguous():
            normalized_ids = normalized_ids.contiguous()
        device = normalized_ids.device
        if fused_ids.numel() == 0:
            empty = torch.empty((0,), dtype=torch.int64, device=device)
            return LookupIdsPayload(
                rank=rank,
                destination_ranks=empty,
                source_ranks=empty,
                row_positions=empty,
                fused_ids=empty,
            )
        row_positions = torch.arange(normalized_ids.numel(), dtype=torch.int64, device=device)
        source_ranks = torch.full(
            (normalized_ids.numel(),),
            fill_value=rank,
            dtype=torch.int64,
            device=device,
        )
        destination_ranks = torch.remainder(normalized_ids, int(world_size))
        return LookupIdsPayload(
            rank=rank,
            destination_ranks=destination_ranks,
            source_ranks=source_ranks,
            row_positions=row_positions,
            fused_ids=normalized_ids,
        )

    def _lookup_fused_embeddings_single_node_distributed(
        self,
        fused_ids: torch.Tensor,
        *,
        compute_device: torch.device,
    ) -> torch.Tensor:
        profile = _new_lookup_profile()
        dist = torch.distributed
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        backend = dist
        self._prepare_single_node_local_shm_fast_path_client(rank)

        local_payload = self._build_lookup_request_payload(
            fused_ids,
            rank=rank,
            world_size=world_size,
        )
        t_exchange_start = perf_counter()
        exchange_ids_start = time.perf_counter()
        gathered_requests = exchange_lookup_ids(
            local_payload,
            world_size=world_size,
            backend=backend,
        )
        self._perf_add("lookup_owner_exchange_ms", (perf_counter() - t_exchange_start) * 1e3)
        profile["lookup_exchange_ids_ms"] += (
            time.perf_counter() - exchange_ids_start
        ) * 1e3

        owner_source_rank_tensors = [
            payload.source_ranks for payload in gathered_requests if payload.source_ranks.numel() > 0
        ]
        owner_row_position_tensors = [
            payload.row_positions for payload in gathered_requests if payload.row_positions.numel() > 0
        ]
        owner_id_tensors = [
            payload.fused_ids for payload in gathered_requests if payload.fused_ids.numel() > 0
        ]
        payload_device = local_payload.fused_ids.device

        if owner_id_tensors:
            local_ids = torch.cat(owner_id_tensors, dim=0).contiguous()
            t_lookup_start = perf_counter()
            lookup_start = time.perf_counter()
            local_embeddings = self.kv_client.local_lookup_flat(
                self._master_config.name,
                local_ids,
            )
            self._perf_add("lookup_local_lookup_ms", (perf_counter() - t_lookup_start) * 1e3)
            profile["lookup_local_lookup_ms"] += (
                time.perf_counter() - lookup_start
            ) * 1e3
            _merge_profile_values(
                profile,
                getattr(self.kv_client, "get_last_local_shm_lookup_profile", lambda: {})(),
            )
        else:
            local_ids = torch.empty((0,), dtype=torch.int64, device=payload_device)
            local_embeddings = torch.empty(
                (0, self._master_config.embedding_dim),
                dtype=torch.float32,
                device=payload_device,
            )
        if local_embeddings.device != compute_device:
            local_embeddings = local_embeddings.to(compute_device)

        response_payload = LookupEmbeddingResponsePayload(
            rank=rank,
            requestor_ranks=(
                torch.cat(owner_source_rank_tensors, dim=0).contiguous()
                if owner_source_rank_tensors
                else torch.empty((0,), dtype=torch.int64, device=local_ids.device)
            ),
            row_positions=(
                torch.cat(owner_row_position_tensors, dim=0).contiguous()
                if owner_row_position_tensors
                else torch.empty((0,), dtype=torch.int64, device=local_ids.device)
            ),
            embeddings=local_embeddings,
        )
        exchange_responses_start = time.perf_counter()
        gathered_responses = exchange_lookup_embedding_responses(
            response_payload,
            world_size=world_size,
            backend=backend,
        )
        profile["lookup_exchange_responses_ms"] += (
            time.perf_counter() - exchange_responses_start
        ) * 1e3
        rebuild_start = time.perf_counter()
        t_reassemble_start = perf_counter()
        rebuilt = reassemble_lookup_embedding_responses(
            gathered_responses,
            requestor_rank=rank,
            total_rows=int(fused_ids.numel()),
        )
        self._perf_add("lookup_reassemble_ms", (perf_counter() - t_reassemble_start) * 1e3)
        profile["lookup_rebuild_ms"] += (time.perf_counter() - rebuild_start) * 1e3
        if rebuilt.device != compute_device:
            transfer_start = time.perf_counter()
            rebuilt = rebuilt.to(compute_device)
            profile["lookup_post_rebuild_h2d_ms"] += (
                time.perf_counter() - transfer_start
            ) * 1e3
        setattr(self, "_single_node_forward_profile", profile)
        return rebuilt

    def _lookup_fused_embeddings_shared_local_shm_single_table(
        self,
        fused_ids: torch.Tensor,
        *,
        compute_device: torch.device,
    ) -> torch.Tensor:
        profile = _new_lookup_profile()
        dist = torch.distributed
        dist_available = (
            not hasattr(dist, "is_available") or bool(dist.is_available())
        )
        dist_initialized = (
            hasattr(dist, "is_initialized") and bool(dist.is_initialized())
        )
        rank = int(dist.get_rank()) if dist_available and dist_initialized else 0
        self._prepare_single_node_local_shm_fast_path_client(rank)
        lookup_start = time.perf_counter()
        local_embeddings = self.kv_client.local_lookup_flat(
            self._master_config.name,
            fused_ids,
        )
        profile["lookup_local_lookup_ms"] += (time.perf_counter() - lookup_start) * 1e3
        _merge_profile_values(
            profile,
            getattr(self.kv_client, "get_last_local_shm_lookup_profile", lambda: {})(),
        )
        if local_embeddings.device != compute_device:
            transfer_start = time.perf_counter()
            local_embeddings = local_embeddings.to(compute_device)
            profile["lookup_post_rebuild_h2d_ms"] += (
                time.perf_counter() - transfer_start
            ) * 1e3
        setattr(self, "_single_node_forward_profile", profile)
        return local_embeddings

    def _lookup_fused_embeddings_gpu_cache(
        self,
        fused_ids: torch.Tensor,
        *,
        compute_device: torch.device,
    ) -> torch.Tensor:
        """Forward lookup backed by the GPU cache (any backend).

        Serves hits from the GPU cache; misses are fetched via the active
        backend (BRPC/GRPC/RDMA) and filled back into the cache.  This is
        the path used by the BagPipe controller when the local_shm fast
        path is unavailable, so the GPU cache is actually queried instead
        of bypassed by a plain pull.
        """
        t_lookup = perf_counter()
        embedding_dim = self._master_config.embedding_dim
        ids_for_query = fused_ids
        if ids_for_query.device.type != "cuda":
            ids_for_query = ids_for_query.to(compute_device)
        if not ids_for_query.is_contiguous():
            ids_for_query = ids_for_query.contiguous()
        embeddings = self.kv_client.gpu_cache_lookup_flat(
            ids_for_query, embedding_dim
        )
        if embeddings.device != compute_device:
            embeddings = embeddings.to(compute_device)
        self._perf_add("lookup_gpu_cache_ms", (perf_counter() - t_lookup) * 1e3)
        return embeddings

    def _prefill_gpu_cache_from_fused_prefetch(
        self,
        fused_values_all: torch.Tensor,
        *,
        compute_device: torch.device,
    ) -> bool:
        if self._fused_prefetch_handle is None:
            return False
        allow_cpu_prefill = bool(getattr(self.kv_client, "allow_cpu_gpu_cache_prefill", False))
        if compute_device.type != "cuda" and not allow_cpu_prefill:
            self._perf_add("planned_gpu_cache_prefill_no_cuda", 1.0)
            self._perf_add("planned_gpu_cache_prefill_fallbacks", 1.0)
            self._pop_current_fused_prefetch_slot()
            return False
        prefill = self._gpu_cache_prefill_api()
        if prefill is None:
            self._pop_current_fused_prefetch_slot()
            return False
        ids_cached = self._fused_ids_cpu
        if ids_cached is None:
            self._perf_add("planned_gpu_cache_prefill_fallbacks", 1.0)
            self._pop_current_fused_prefetch_slot()
            return False
        t_prefill_start = perf_counter()
        prefill_ids = ids_cached.to(device=compute_device, dtype=torch.int64)
        if not prefill_ids.is_contiguous():
            prefill_ids = prefill_ids.contiguous()
        try:
            t_wait_start = perf_counter()
            all_embeddings = self.kv_client.wait_and_get(
                self._fused_prefetch_handle,
                self._master_config.embedding_dim,
                device=compute_device,
            )
            t_wait_end = perf_counter()
            self._perf_add("planned_gpu_cache_prefill_wait_ms", (t_wait_end - t_wait_start) * 1e3)
            self._prefetch_wait_latencies.append(t_wait_end - t_wait_start)
            issue_latency = t_wait_start - (self._fused_prefetch_issue_ts or t_wait_start)
            self._prefetch_issue_latencies.append(issue_latency)
            if all_embeddings.size(0) == ids_cached.numel():
                prefill_values = all_embeddings
            elif all_embeddings.size(0) == fused_values_all.numel():
                prefill_ids = fused_values_all.to(device=compute_device, dtype=torch.int64)
                prefill_values = all_embeddings
            else:
                logging.warning(
                    "[EBC] Fused prefetch prefill size mismatch: got %s rows, "
                    "expected %s unique rows or %s fused rows; falling back to "
                    "local lookup prefill.",
                    all_embeddings.size(0),
                    ids_cached.numel(),
                    fused_values_all.numel(),
                )
                self._perf_add("planned_gpu_cache_prefill_result_size_mismatches", 1.0)
                self._perf_add("planned_gpu_cache_prefill_fallbacks", 1.0)
                prefill_values = self.kv_client.local_lookup_flat(
                    self._master_config.name,
                    prefill_ids,
                )
        except Exception as exc:
            logging.warning(
                "[EBC] Fused prefetch wait failed during GPU cache prefill; "
                "falling back to local lookup prefill: %s",
                exc,
            )
            self._perf_add("planned_gpu_cache_prefill_wait_failures", 1.0)
            self._perf_add("planned_gpu_cache_prefill_fallbacks", 1.0)
            prefill_values = self.kv_client.local_lookup_flat(
                self._master_config.name,
                prefill_ids,
            )
        if not prefill_values.is_contiguous():
            prefill_values = prefill_values.contiguous()
        prefill(self._master_config.name, prefill_ids, prefill_values)
        self._perf_add("planned_gpu_cache_prefill_ms", (perf_counter() - t_prefill_start) * 1e3)
        self._perf_add("planned_gpu_cache_prefill_batches", 1.0)
        self._perf_add("planned_gpu_cache_prefill_ids", float(prefill_ids.numel()))
        self._perf_add("planned_gpu_cache_prefill_successes", 1.0)
        self._pop_current_fused_prefetch_slot()
        return True

    def _consume_fused_prefetch_embeddings(
        self,
        fused_values_all: torch.Tensor,
        cpu_ids: torch.Tensor,
        *,
        compute_device: torch.device,
    ) -> tuple[torch.Tensor, bool]:
        t_wait_start = perf_counter()
        t_wait_perf_start = perf_counter()
        all_embeddings = self.kv_client.wait_and_get(
            self._fused_prefetch_handle,
            self._master_config.embedding_dim,
            device=compute_device,
        )
        t_wait_end = perf_counter()
        wait_latency = t_wait_end - t_wait_start
        self._prefetch_wait_latencies.append(wait_latency)
        issue_latency = t_wait_start - (self._fused_prefetch_issue_ts or t_wait_start)
        self._prefetch_issue_latencies.append(issue_latency)
        self._perf_add("lookup_wait_ms", (perf_counter() - t_wait_perf_start) * 1e3)

        used_fused_prefetch = True
        invalid_ids = self._fused_invalid_ids_cpu
        if all_embeddings.size(0) != fused_values_all.numel():
            inv = self._fused_inverse
            ids_cached = self._fused_ids_cpu
            if inv is not None and ids_cached is not None and all_embeddings.size(0) == ids_cached.numel():
                current_unique_ids = torch.unique(fused_values_all.detach().to(dtype=torch.int64, device="cpu"))
                ids_match_current_batch = (
                    ids_cached.detach().to(dtype=torch.int64, device="cpu").flatten().numel()
                    == current_unique_ids.numel()
                    and torch.equal(
                        ids_cached.detach().to(dtype=torch.int64, device="cpu").flatten(),
                        current_unique_ids,
                    )
                )
                if (
                    ids_match_current_batch
                    and (invalid_ids is None or invalid_ids.numel() == 0)
                ):
                    indexer = inv.to(device=all_embeddings.device, dtype=torch.long)
                    all_embeddings = all_embeddings.index_select(0, indexer)
                else:
                    all_embeddings = self._merge_partial_fused_prefetch_embeddings(
                        fused_values_all,
                        prefetched_ids=ids_cached,
                        prefetched_embeddings=all_embeddings,
                        invalid_ids=invalid_ids,
                        compute_device=compute_device,
                    )
            else:
                unique_ids, inverse = torch.unique(fused_values_all, return_inverse=True)
                if all_embeddings.size(0) == unique_ids.size(0) and (
                    invalid_ids is None or invalid_ids.numel() == 0
                ):
                    all_embeddings = all_embeddings.index_select(0, inverse)
                else:
                    logging.warning(
                        "[EBC] Fused prefetch result size mismatch: got %s, "
                        "expected %s, falling back to pull.",
                        all_embeddings.size(0),
                        fused_values_all.numel(),
                    )
                    all_embeddings = self._pull_fused_rows(
                        fused_values_all,
                        target_device=compute_device,
                    )
                    used_fused_prefetch = False
        elif invalid_ids is not None and invalid_ids.numel() > 0:
            unique_ids, inverse = torch.unique(
                fused_values_all.detach().to(dtype=torch.int64, device="cpu"),
                sorted=True,
                return_inverse=True,
            )
            if all_embeddings.size(0) == unique_ids.size(0):
                all_embeddings = self._merge_partial_fused_prefetch_embeddings(
                    fused_values_all,
                    prefetched_ids=unique_ids,
                    prefetched_embeddings=all_embeddings,
                    invalid_ids=invalid_ids,
                    compute_device=compute_device,
                )
            else:
                all_embeddings = self._merge_partial_fused_prefetch_embeddings(
                    fused_values_all,
                    prefetched_ids=fused_values_all.detach().to(dtype=torch.int64, device="cpu"),
                    prefetched_embeddings=all_embeddings,
                    invalid_ids=invalid_ids,
                    compute_device=compute_device,
                )
        return all_embeddings, used_fused_prefetch

    def _merge_partial_fused_prefetch_embeddings(
        self,
        fused_values_all: torch.Tensor,
        *,
        prefetched_ids: torch.Tensor,
        prefetched_embeddings: torch.Tensor,
        invalid_ids: torch.Tensor | None = None,
        compute_device: torch.device,
    ) -> torch.Tensor:
        t_merge_start = perf_counter()
        batch_ids_cpu = fused_values_all.detach().to(dtype=torch.int64, device="cpu")
        prefetched_ids_cpu = prefetched_ids.detach().to(dtype=torch.int64, device="cpu")
        unique_batch_ids = torch.unique(batch_ids_cpu, sorted=True)
        if isinstance(invalid_ids, torch.Tensor) and invalid_ids.numel() > 0:
            invalid_cpu = invalid_ids.detach().to(dtype=torch.int64, device="cpu").flatten()
            valid_prefetched_mask = ~torch.isin(prefetched_ids_cpu, invalid_cpu)
            prefetched_ids_cpu = prefetched_ids_cpu[valid_prefetched_mask]
            prefetched_embeddings = prefetched_embeddings.index_select(
                0,
                valid_prefetched_mask.nonzero(as_tuple=False).flatten().to(
                    device=prefetched_embeddings.device,
                    dtype=torch.long,
                ),
            )

        prefetched_ids_cpu, sort_index = torch.sort(prefetched_ids_cpu)
        if sort_index.numel() > 0:
            prefetched_embeddings = prefetched_embeddings.index_select(
                0,
                sort_index.to(device=prefetched_embeddings.device, dtype=torch.long),
            )

        has_prefetched = (
            torch.isin(unique_batch_ids, prefetched_ids_cpu)
            if prefetched_ids_cpu.numel() > 0
            else torch.zeros_like(unique_batch_ids, dtype=torch.bool)
        )
        missing_ids_cpu = unique_batch_ids[~has_prefetched]
        self._perf_add("lookup_partial_prefetched_ids", float(prefetched_ids_cpu.numel()))
        self._perf_add("lookup_partial_missing_ids", float(missing_ids_cpu.numel()))

        all_ids_cpu = prefetched_ids_cpu
        all_embeddings = prefetched_embeddings
        if missing_ids_cpu.numel() > 0:
            missing_tensor = missing_ids_cpu.to(device=fused_values_all.device)
            missing_embeddings = self._lookup_missing_fused_prefetch_rows(missing_tensor)
            if missing_embeddings.device != prefetched_embeddings.device:
                missing_embeddings = missing_embeddings.to(prefetched_embeddings.device)
            if missing_embeddings.dtype != prefetched_embeddings.dtype:
                missing_embeddings = missing_embeddings.to(dtype=prefetched_embeddings.dtype)
            all_ids_cpu = torch.cat([all_ids_cpu, missing_ids_cpu])
            all_embeddings = torch.cat([all_embeddings, missing_embeddings], dim=0)

        all_ids_cpu, all_sort_index = torch.sort(all_ids_cpu)
        all_embeddings = all_embeddings.index_select(
            0,
            all_sort_index.to(device=all_embeddings.device, dtype=torch.long),
        )
        positions = torch.searchsorted(all_ids_cpu, batch_ids_cpu)
        merged = all_embeddings.index_select(
            0,
            positions.to(device=all_embeddings.device, dtype=torch.long),
        )
        if merged.device != compute_device:
            merged = merged.to(compute_device)
        self._perf_add("lookup_partial_merge_ms", (perf_counter() - t_merge_start) * 1e3)
        return merged

    def _lookup_missing_fused_prefetch_rows(self, ids: torch.Tensor) -> torch.Tensor:
        return self._pull_fused_rows(
            ids,
            target_device=ids.device,
            try_local_lookup=True,
        )

    def _pull_fused_rows(
        self,
        ids: torch.Tensor,
        *,
        target_device: torch.device,
        metric_key: str = "lookup_gpu_cache_pull_ms",
        try_local_lookup: bool = False,
    ) -> torch.Tensor:
        if try_local_lookup:
            local_lookup = getattr(self.kv_client, "local_lookup_flat", None)
            if callable(local_lookup):
                try:
                    t_lookup_start = perf_counter()
                    rows = local_lookup(self._master_config.name, ids)
                    self._perf_add(
                        "lookup_local_lookup_ms", (perf_counter() - t_lookup_start) * 1e3
                    )
                    _merge_profile_values(
                        getattr(self, "_single_node_forward_profile", {}),
                        getattr(
                            self.kv_client, "get_last_local_shm_lookup_profile", lambda: {}
                        )(),
                    )
                    return rows
                except RuntimeError as exc:
                    logging.debug(
                        "[EBC] local lookup unavailable for partial fused prefetch merge; "
                        "trying GPU-cache-aware pull for %s ids: %s",
                        ids.numel(),
                        exc,
                    )
        gpu_cache_pull = getattr(self.kv_client, "pull_with_gpu_cache", None)
        if callable(gpu_cache_pull):
            try:
                t_gpu_cache_pull_start = perf_counter()
                rows = gpu_cache_pull(name=self._master_config.name, ids=ids)
                self._perf_add(
                    metric_key,
                    (perf_counter() - t_gpu_cache_pull_start) * 1e3,
                )
                if rows.device != target_device:
                    rows = rows.to(target_device)
                return rows
            except RuntimeError as exc:
                logging.debug(
                    "[EBC] GPU-cache-aware fused pull unavailable; falling back "
                    "to CPU pull for %s ids: %s",
                    ids.numel(),
                    exc,
                )
        t_pull_start = perf_counter()
        rows = self.kv_client.pull(name=self._master_config.name, ids=ids.to("cpu"))
        self._perf_add("lookup_fallback_pull_ms", (perf_counter() - t_pull_start) * 1e3)
        if rows.device != target_device:
            rows = rows.to(target_device)
        return rows

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        setattr(self, "_single_node_forward_profile", {})
        t_lookup_total_start = perf_counter()
        try:
            return self._forward_impl(features)
        finally:
            self._perf_add("lookup_total_ms", (perf_counter() - t_lookup_total_start) * 1e3)

    def _forward_impl(self, features: KeyedJaggedTensor) -> KeyedTensor:
        # Determine if we can enable fused single-call path safely
        keys_in_batch = list(features.keys())
        dims_this_batch: List[int] = [
            self._config_by_feature[k].embedding_dim for k in keys_in_batch
        ]
        can_uniform_dim = all(d == dims_this_batch[0] for d in dims_this_batch) if len(dims_this_batch) > 0 else False

        use_fused = (
            self._enable_fusion
            and self._master_config is not None
            and can_uniform_dim
        )

        if use_fused:
            # Build fused global IDs with bit-prefix encoding: fused_id = (table_id << k) | id
            fused_values_list: List[torch.Tensor] = []
            lengths_total_list: List[torch.Tensor] = []
            compute_device = features.device()  # target device (cuda or cpu)
            uses_shared_local_shm = self._uses_shared_local_shm_single_table()
            use_shared_local_shm_direct_fast_path = (
                self._can_use_single_node_distributed_fast_path() and uses_shared_local_shm
            )
            use_local_shm_direct_fast_path = uses_shared_local_shm
            use_single_node_owner_exchange_fast_path = (
                self._can_use_single_node_distributed_fast_path()
                and not use_shared_local_shm_direct_fast_path
            )
            use_single_node_fast_path = (
                use_local_shm_direct_fast_path
                or
                use_shared_local_shm_direct_fast_path
                or use_single_node_owner_exchange_fast_path
            )

            for key in keys_in_batch:
                kjt_per_feature = features[key]
                values = kjt_per_feature.values()
                lengths = kjt_per_feature.lengths()
                # ensure dtype int64
                if values.dtype != torch.int64:
                    values = values.to(torch.int64)

                if values.numel() > 0:
                    table_idx = self._feature_table_indices[key]
                    prefix = (table_idx << self._fusion_k)
                    fused_values = values + prefix
                    fused_values_list.append(fused_values)
                # Even if empty, we should keep lengths for offsets construction
                lengths_total_list.append(lengths)

            if len(fused_values_list) > 0:
                fused_values_all = torch.cat(fused_values_list, dim=0).to(dtype=torch.int64)
            else:
                fused_values_all = torch.empty((0,), dtype=torch.int64, device=compute_device)
            if not fused_values_all.is_contiguous():
                fused_values_all = fused_values_all.contiguous()
            if fused_values_all.device != compute_device:
                fused_values_all = fused_values_all.to(compute_device)
            cpu_ids = (
                fused_values_all.to("cpu")
                if fused_values_all.device.type != "cpu"
                else fused_values_all
            )
            if use_single_node_fast_path:
                trace_ids = fused_values_all
            else:
                trace_ids = cpu_ids

            lengths_total = torch.cat(lengths_total_list, dim=0) if len(lengths_total_list) > 0 else torch.empty((0,), dtype=torch.int32, device=compute_device)

            # Obtain embeddings: prefer single-node owner lookup; else fused prefetch; else merge per-feature prefetch; else single pull
            all_embeddings: torch.Tensor
            used_fused_prefetch = False
            _gpu_cache_fn = getattr(self.kv_client, "is_gpu_cache_enabled", None)
            _gpu_cache_on = bool(_gpu_cache_fn()) if callable(_gpu_cache_fn) else False
            if use_local_shm_direct_fast_path:
                is_gpu_cache_enabled = getattr(self.kv_client, "is_gpu_cache_enabled", None)
                gpu_cache_enabled = bool(is_gpu_cache_enabled()) if callable(is_gpu_cache_enabled) else True
                if self._fused_prefetch_handle is not None and not gpu_cache_enabled:
                    all_embeddings, used_fused_prefetch = self._consume_fused_prefetch_embeddings(
                        fused_values_all,
                        cpu_ids,
                        compute_device=compute_device,
                    )
                else:
                    self._prefill_gpu_cache_from_fused_prefetch(
                        fused_values_all,
                        compute_device=compute_device,
                    )
                    all_embeddings = self._lookup_fused_embeddings_shared_local_shm_single_table(
                        fused_values_all,
                        compute_device=compute_device,
                    )
            elif use_single_node_owner_exchange_fast_path:
                all_embeddings = self._lookup_fused_embeddings_single_node_distributed(
                    fused_values_all,
                    compute_device=compute_device,
                )
            elif self._fused_prefetch_handle is not None:
                all_embeddings, used_fused_prefetch = self._consume_fused_prefetch_embeddings(
                    fused_values_all,
                    cpu_ids,
                    compute_device=compute_device,
                )
            elif len(self._prefetch_handles) > 0:
                # Gather per-feature prefetched embeddings in the same order
                per_feature_embs: List[torch.Tensor] = []
                for key in keys_in_batch:
                    kjt_per_feature = features[key]
                    values = kjt_per_feature.values()
                    if key in self._prefetch_handles:
                        emb = self._consume_feature_prefetch_handle(
                            key, values, record_lookup_wait=True
                        )
                        if emb.size(0) != values.numel():
                            logging.warning(
                                f"[EBC] Prefetch result size mismatch for feature '{key}': got {emb.size(0)}, expected {values.numel()}, falling back to pull."
                            )
                            table_idx = self._feature_table_indices[key]
                            fused_ids_local = values.to(torch.int64) + (table_idx << self._fusion_k)
                            emb = self._pull_fused_rows(
                                fused_ids_local,
                                target_device=values.device,
                            )
                        per_feature_embs.append(emb)
                    else:
                        # No prefetch for this feature, do a synchronous pull with fused IDs for that feature only
                        if values.numel() == 0:
                            config = self._config_for_feature(key)
                            per_feature_embs.append(torch.empty((0, config.embedding_dim), device=features.device(), dtype=torch.float32))
                        else:
                            table_idx = self._feature_table_indices[key]
                            prefix = (table_idx << self._fusion_k)
                            fused_ids_local = values.to(torch.int64) + prefix
                            emb = self._pull_fused_rows(
                                fused_ids_local,
                                target_device=values.device,
                            )
                            per_feature_embs.append(emb)
                all_embeddings = torch.cat(per_feature_embs, dim=0) if len(per_feature_embs) > 0 else torch.empty((0, self._master_config.embedding_dim), device=features.device(), dtype=torch.float32)
            elif _gpu_cache_on:
                all_embeddings = self._lookup_fused_embeddings_gpu_cache(
                    fused_values_all,
                    compute_device=compute_device,
                )
            else:
                # Single pull for all fused IDs
                all_embeddings = self._pull_fused_rows(
                    fused_values_all,
                    target_device=compute_device,
                )
            all_embeddings.requires_grad_()

            def grad_hook_fused(grad, ids=trace_ids, master_name=self._master_config.name):
                self._append_trace(master_name, ids, grad)

            all_embeddings.register_hook(grad_hook_fused)

            # Pool across all bags (feature-major order)
            pool_device = all_embeddings.device
            local_indices = torch.arange(len(fused_values_all), device=pool_device, dtype=torch.long)
            offsets = torch.cat([
                torch.tensor([0], device=pool_device),
                torch.cumsum(lengths_total.to(device=pool_device), 0)[:-1]
                if lengths_total.numel() > 0
                else torch.empty((0,), device=pool_device, dtype=lengths_total.dtype)
            ])
            t_pool_start = perf_counter()
            pooled_total = F.embedding_bag(
                input=local_indices,
                weight=all_embeddings,
                offsets=offsets,
                mode="sum",
                sparse=False,
            )
            self._perf_add("pool_embedding_bag_ms", (perf_counter() - t_pool_start) * 1e3)

            # Split back by feature (each has B_i bags = lengths.size(0))
            pooled_embs_list: List[torch.Tensor] = []
            split_sizes = [l.numel() for l in lengths_total_list]
            if sum(split_sizes) > 0:
                pieces = torch.split(pooled_total, split_sizes, dim=0)
            else:
                pieces = [torch.empty((0, dims_this_batch[0]), device=pooled_total.device, dtype=pooled_total.dtype) for _ in split_sizes]
            for piece in pieces:
                pooled_embs_list.append(piece)

            concatenated_embs = torch.cat(pooled_embs_list, dim=1) if len(pooled_embs_list) > 0 else torch.empty((0, dims_this_batch[0]), device=compute_device, dtype=torch.float32)

            length_per_key = dims_this_batch

            out = KeyedTensor(
                keys=keys_in_batch,
                values=concatenated_embs,
                length_per_key=length_per_key,
            )
            # Clear fused prefetch handle after consumption
            if used_fused_prefetch:
                self._pop_current_fused_prefetch_slot()
            return out

        # Fallback: per-feature path (original behavior)
        pooled_embs_list = []
        for key in keys_in_batch:
            config_name = self._config_names[key]
            kjt_per_feature = features[key]
            values = kjt_per_feature.values()
            lengths = kjt_per_feature.lengths()

            if values.numel() == 0:
                config = self._config_for_feature(key)
                pooled_embs = torch.zeros(len(lengths), config.embedding_dim, device=features.device(), dtype=torch.float32)
            else:
                # Prefer prefetched embeddings if provided for this feature
                ids_used = values  # default
                if key in self._prefetch_handles:
                    all_embeddings = self._consume_feature_prefetch_handle(key, values)
                    if all_embeddings.size(0) != values.numel():
                        logging.warning(
                            f"[EBC] Prefetch result size mismatch for feature '{key}': got {all_embeddings.size(0)}, expected {values.numel()}, falling back to pull."
                        )
                        # If fusion was enabled at init, IDs in storage are prefixed; add offset here
                        if self._enable_fusion:
                            table_idx = self._feature_table_indices[key]
                            prefix = (table_idx << self._fusion_k)
                            ids_used = values.to(torch.int64) + prefix
                            cpu_ids_local = ids_used.to('cpu')
                            t_pull_start = perf_counter()
                            all_embeddings = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids_local)
                            self._perf_add("lookup_fallback_pull_ms", (perf_counter() - t_pull_start) * 1e3)
                            if values.device.type == 'cuda':
                                all_embeddings = all_embeddings.to(values.device)
                        else:
                            ids_used = values
                            t_pull_start = perf_counter()
                            all_embeddings = self.kv_client.pull(name=config_name, ids=ids_used)
                            self._perf_add("lookup_fallback_pull_ms", (perf_counter() - t_pull_start) * 1e3)
                            if values.device.type == 'cuda':
                                all_embeddings = all_embeddings.to(values.device)
                else:
                    if self._enable_fusion:
                        table_idx = self._feature_table_indices[key]
                        prefix = (table_idx << self._fusion_k)
                        ids_used = values.to(torch.int64) + prefix
                        cpu_ids_local = ids_used.to('cpu')
                        t_pull_start = perf_counter()
                        all_embeddings = self.kv_client.pull(name=self._master_config.name, ids=cpu_ids_local)
                        self._perf_add("lookup_fallback_pull_ms", (perf_counter() - t_pull_start) * 1e3)
                        if values.device.type == 'cuda':
                            all_embeddings = all_embeddings.to(values.device)
                    else:
                        ids_used = values
                        t_pull_start = perf_counter()
                        all_embeddings = self.kv_client.pull(name=config_name, ids=ids_used)
                        self._perf_add("lookup_fallback_pull_ms", (perf_counter() - t_pull_start) * 1e3)
                        if values.device.type == 'cuda':
                            all_embeddings = all_embeddings.to(values.device)
                all_embeddings.requires_grad_()

                def grad_hook(grad, name=(self._master_config.name if self._enable_fusion else config_name), ids=ids_used):
                    self._append_trace(name, ids, grad)
                all_embeddings.register_hook(grad_hook)

                local_indices = torch.arange(len(values), device=values.device, dtype=torch.long)
                offsets = torch.cat([torch.tensor([0], device=lengths.device), torch.cumsum(lengths, 0)[:-1]])
                t_pool_start = perf_counter()
                pooled_embs = F.embedding_bag(
                    input=local_indices,
                    weight=all_embeddings,
                    offsets=offsets,
                    mode="sum",
                    sparse=False,
                )
                self._perf_add("pool_embedding_bag_ms", (perf_counter() - t_pool_start) * 1e3)
            pooled_embs_list.append(pooled_embs)

        concatenated_embs = torch.cat(pooled_embs_list, dim=1)
        length_per_key = [
            self._config_by_feature[key].embedding_dim for key in keys_in_batch
        ]
        return KeyedTensor(
            keys=keys_in_batch,
            values=concatenated_embs,
            length_per_key=length_per_key,
        )

    def report_prefetch_stats(self, reset: bool = True) -> Dict[str, float]:
        n = len(self._prefetch_wait_latencies)
        if n == 0:
            stats = {
                "batches_prefetched": 0,
                "avg_wait_ms": 0.0,
                "avg_issue_to_wait_ms": 0.0,
                "total_prefetched_ids": self._prefetch_total_ids,
                "embeddings_per_sec_during_wait": 0.0,
            }
        else:
            avg_wait = sum(self._prefetch_wait_latencies) / n
            avg_issue = sum(self._prefetch_issue_latencies) / n if self._prefetch_issue_latencies else 0.0
            stats = {
                "batches_prefetched": n,
                "avg_wait_ms": avg_wait * 1000.0,
                "avg_issue_to_wait_ms": avg_issue * 1000.0,
                "total_prefetched_ids": self._prefetch_total_ids,
                "embeddings_per_sec_during_wait": (self._prefetch_total_ids / sum(self._prefetch_wait_latencies)) if sum(self._prefetch_wait_latencies) > 0 else 0.0,
            }
        if reset:
            self._prefetch_issue_ts.clear()
            self._prefetch_sizes.clear()
            self._prefetch_wait_latencies.clear()
            self._prefetch_issue_latencies.clear()
            self._prefetch_total_ids = 0
        return stats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tables={self.feature_keys})"
