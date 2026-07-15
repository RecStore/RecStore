import time

import torch
from typing import List, Union, Dict, Tuple, Any
from time import perf_counter
from .single_node_exchange import SparseGradPayload, exchange_sparse_grads

_LOCAL_FAST_PATH_BACKENDS = {"local_shm", "hierkv"}

class DistEmbedding:
    pass

def _get_kv_client_if_needed(params: List[Any]):
    """Dynamically imports and returns the KV client if params are provided."""
    if params:
        for mod in params:
            module_kv_client = getattr(mod, "kv_client", None)
            if module_kv_client is not None:
                return module_kv_client
        from .KVClient import get_kv_client
        from .DistEmb import DistEmbedding as DistEmbeddingImpl
        global DistEmbedding
        DistEmbedding = DistEmbeddingImpl
        return get_kv_client()
    return None

def _process_dist_embedding_module(mod: DistEmbedding, lr: float):
    """Handles the optimization step for a DistEmbedding module using gradient accumulation."""
    if not mod._trace:
        return

    all_ids = torch.cat([ids for ids, _ in mod._trace])
    all_grads = torch.cat([grads for _, grads in mod._trace])

    unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)

    summed_grads = torch.zeros(
        (len(unique_ids), mod.embedding_dim),
        device=all_grads.device,
        dtype=all_grads.dtype
    )
    summed_grads.index_add_(0, inverse_indices, all_grads)

    scaled_grads = lr * summed_grads
    
    current_weights = mod.weight[unique_ids]
    updated_weights = current_weights - scaled_grads
    mod.weight[unique_ids] = updated_weights

def _process_generic_module_with_trace(mod: Any, lr: float, kv_client: Any):
    """Handles sparse trace aggregation and backend updates for generic modules."""
    if not mod._trace:
        return []

    traces_by_name: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for entry in mod._trace:
        if isinstance(entry, dict):
            name = entry["name"]
            ids = entry["ids"]
            if "grads" in entry:
                grads = entry["grads"]
            else:
                grads = entry["grad"].unsqueeze(0).expand(int(entry["count"]), -1)
        else:
            name, ids, grads = entry
        traces_by_name.setdefault(name, []).append((ids, grads))

    module_kv_client = getattr(mod, "kv_client", None) or kv_client
    handles = []
    for name, entries in traces_by_name.items():
        all_ids = torch.cat([ids for ids, _ in entries], dim=0)
        all_grads = torch.cat([grads for _, grads in entries], dim=0)

        unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)
        summed_grads = torch.zeros(
            (len(unique_ids), all_grads.size(1)),
            device=all_grads.device,
            dtype=all_grads.dtype,
        )
        summed_grads.index_add_(0, inverse_indices, all_grads)

        # Backend sparse optimizers own learning-rate application for these modules.
        handle = module_kv_client.update_async(name=name, ids=unique_ids, grads=summed_grads)
        handles.append(
            (
                module_kv_client,
                handle,
                {
                    "module": mod,
                    "name": name,
                    "ids": unique_ids.detach(),
                    "grads": summed_grads.detach(),
                    "lr": float(lr),
                },
            )
        )
    return handles


def _can_use_single_node_distributed_fast_path(mod: Any) -> bool:
    if not getattr(mod, "enable_single_node_distributed_fast_path", False):
        return False
    if getattr(mod, "single_node_distributed_mode", None) != "single_node":
        return False
    if getattr(mod, "single_node_owner_policy", None) != "hash_mod_world_size":
        return False
    if getattr(mod, "single_node_ps_backend", None) not in _LOCAL_FAST_PATH_BACKENDS:
        return False
    dist = getattr(torch, "distributed", None)
    if dist is None or not hasattr(dist, "is_initialized") or not dist.is_initialized():
        return False
    if not hasattr(dist, "get_world_size") or dist.get_world_size() <= 1:
        return False
    return True


def _uses_shared_local_shm_single_table(mod: Any) -> bool:
    if not getattr(mod, "_enable_fusion", False):
        return False
    if getattr(mod, "_master_config", None) is None:
        return False
    module_kv_client = getattr(mod, "kv_client", None)
    if module_kv_client is None:
        return False
    probe = getattr(module_kv_client, "is_shared_local_shm_table", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def _can_use_shared_local_shm_direct_fast_path(mod: Any) -> bool:
    return _uses_shared_local_shm_single_table(mod)


def _collect_traces_by_name(mod: Any) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    traces_by_name: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for entry in mod._trace:
        if isinstance(entry, dict):
            name = entry["name"]
            ids = entry["ids"]
            if "grads" in entry:
                grads = entry["grads"]
            else:
                grads = entry["grad"].unsqueeze(0).expand(int(entry["count"]), -1)
        else:
            name, ids, grads = entry
        traces_by_name.setdefault(name, []).append((ids, grads))
    return traces_by_name


def _aggregate_ids_and_grads(
    ids: torch.Tensor,
    grads: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
    summed_grads = torch.zeros(
        (len(unique_ids), grads.size(1)),
        device=grads.device,
        dtype=grads.dtype,
    )
    summed_grads.index_add_(0, inverse_indices, grads)
    return unique_ids, summed_grads


def _merge_numeric_profile(dst: Dict[str, float], src: Dict[str, Any] | None) -> None:
    if not isinstance(src, dict):
        return
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0.0) + float(value)


def _process_generic_module_with_trace_single_node_distributed(
    mod: Any,
    lr: float,
) -> List[Dict[str, Any]]:
    if not mod._trace:
        return []

    dist = torch.distributed
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    exchange_backend = dist
    module_kv_client = getattr(mod, "kv_client", None)
    if module_kv_client is None:
        raise RuntimeError("single-node distributed sparse update requires module kv_client")
    current_backend = None
    if hasattr(module_kv_client, "current_ps_backend"):
        current_backend = module_kv_client.current_ps_backend()
    target_backend = getattr(mod, "single_node_ps_backend", "local_shm")
    if hasattr(module_kv_client, "activate_shard"):
        module_kv_client.activate_shard(rank)
    if current_backend != target_backend:
        if hasattr(module_kv_client, "set_ps_backend"):
            module_kv_client.set_ps_backend(target_backend)

    profile = {
        "trace_collect_ms": 0.0,
        "trace_aggregate_ms": 0.0,
        "exchange_ms": 0.0,
        "owner_aggregate_ms": 0.0,
        "local_update_ms": 0.0,
    }
    payloads: List[Dict[str, Any]] = []
    collect_start = time.perf_counter()
    traces_by_name = _collect_traces_by_name(mod)
    profile["trace_collect_ms"] += (time.perf_counter() - collect_start) * 1e3
    for name, entries in traces_by_name.items():
        aggregate_start = time.perf_counter()
        all_ids = torch.cat([ids for ids, _ in entries], dim=0)
        all_grads = torch.cat([grads for _, grads in entries], dim=0)
        unique_ids, summed_grads = _aggregate_ids_and_grads(all_ids, all_grads)
        profile["trace_aggregate_ms"] += (time.perf_counter() - aggregate_start) * 1e3
        if getattr(mod, "single_node_owner_policy", "hash_mod_world_size") != "hash_mod_world_size":
            raise RuntimeError("single-node distributed sparse update currently requires hash_mod_world_size")

        normalized_ids = unique_ids.detach().to(dtype=torch.int64)
        normalized_grads = summed_grads.detach().to(dtype=torch.float32)
        destination_ranks = torch.remainder(normalized_ids, world_size)
        payload_device = normalized_ids.device

        local_payload = SparseGradPayload(
            rank=rank,
            destination_ranks=destination_ranks,
            source_ranks=torch.full(
                (normalized_ids.numel(),),
                rank,
                dtype=torch.int64,
                device=payload_device,
            ),
            row_positions=torch.arange(
                normalized_ids.numel(),
                dtype=torch.int64,
                device=payload_device,
            ),
            fused_ids=normalized_ids,
            grads=normalized_grads,
        )
        exchange_start = time.perf_counter()
        gathered_payloads = exchange_sparse_grads(
            local_payload,
            world_size=world_size,
            backend=exchange_backend,
        )
        profile["exchange_ms"] += (time.perf_counter() - exchange_start) * 1e3

        owner_ids: List[torch.Tensor] = []
        owner_grads: List[torch.Tensor] = []
        target_device = None
        aggregate_start = time.perf_counter()
        for payload in gathered_payloads:
            if target_device is None and payload.grads.numel() > 0:
                target_device = payload.grads.device
            if payload.fused_ids.numel() > 0:
                owner_ids.append(payload.fused_ids.detach())
            if payload.grads.numel() > 0:
                owner_grads.append(payload.grads.detach())

        if not owner_ids:
            continue

        if target_device is None:
            target_device = owner_ids[0].device

        owner_ids_tensor = torch.cat(
            [
                ids if ids.device == target_device else ids.to(device=target_device)
                for ids in owner_ids
            ],
            dim=0,
        )
        owner_grads_tensor = torch.cat(
            [
                grads if grads.device == target_device else grads.to(device=target_device)
                for grads in owner_grads
            ],
            dim=0,
        )
        owner_unique_ids, owner_summed_grads = _aggregate_ids_and_grads(
            owner_ids_tensor,
            owner_grads_tensor,
        )
        profile["owner_aggregate_ms"] += (time.perf_counter() - aggregate_start) * 1e3
        local_update_start = time.perf_counter()
        module_kv_client.local_update_flat(
            name=name,
            ids=owner_unique_ids,
            grads=owner_summed_grads,
        )
        payloads.append(
            {
                "module": mod,
                "name": name,
                "ids": owner_unique_ids.detach(),
                "grads": owner_summed_grads.detach(),
                "lr": float(lr),
            }
        )
        profile["local_update_ms"] += (time.perf_counter() - local_update_start) * 1e3
        _merge_numeric_profile(
            profile,
            getattr(module_kv_client, "get_last_local_shm_update_profile", lambda: {})(),
        )

    setattr(mod, "_single_node_fast_path_profile", profile)
    return payloads


def _process_generic_module_with_trace_shared_local_shm_single_table(
    mod: Any,
    lr: float,
) -> List[Dict[str, Any]]:
    if not mod._trace:
        return []

    dist = torch.distributed
    dist_available = (
        not hasattr(dist, "is_available") or bool(dist.is_available())
    )
    dist_initialized = (
        hasattr(dist, "is_initialized") and bool(dist.is_initialized())
    )
    rank = int(dist.get_rank()) if dist_available and dist_initialized else 0
    module_kv_client = getattr(mod, "kv_client", None)
    if module_kv_client is None:
        raise RuntimeError("shared local_shm single-table sparse update requires module kv_client")
    current_backend = None
    if hasattr(module_kv_client, "current_ps_backend"):
        current_backend = module_kv_client.current_ps_backend()
    target_backend = getattr(mod, "single_node_ps_backend", "local_shm")
    if hasattr(module_kv_client, "activate_shard"):
        module_kv_client.activate_shard(rank)
    if current_backend != target_backend:
        if hasattr(module_kv_client, "set_ps_backend"):
            module_kv_client.set_ps_backend(target_backend)

    profile = {
        "trace_collect_ms": 0.0,
        "trace_aggregate_ms": 0.0,
        "exchange_ms": 0.0,
        "owner_aggregate_ms": 0.0,
        "local_update_ms": 0.0,
    }
    payloads: List[Dict[str, Any]] = []
    collect_start = time.perf_counter()
    traces_by_name = _collect_traces_by_name(mod)
    profile["trace_collect_ms"] += (time.perf_counter() - collect_start) * 1e3
    for name, entries in traces_by_name.items():
        aggregate_start = time.perf_counter()
        all_ids = torch.cat([ids for ids, _ in entries], dim=0)
        all_grads = torch.cat([grads for _, grads in entries], dim=0)
        local_unique_ids, local_summed_grads = _aggregate_ids_and_grads(all_ids, all_grads)
        profile["trace_aggregate_ms"] += (time.perf_counter() - aggregate_start) * 1e3
        local_update_start = time.perf_counter()
        module_kv_client.local_update_flat(
            name=name,
            ids=local_unique_ids,
            grads=local_summed_grads,
        )
        payloads.append(
            {
                "module": mod,
                "name": name,
                "ids": local_unique_ids.detach(),
                "grads": local_summed_grads.detach(),
                "lr": float(lr),
            }
        )
        profile["local_update_ms"] += (time.perf_counter() - local_update_start) * 1e3
        _merge_numeric_profile(
            profile,
            getattr(module_kv_client, "get_last_local_shm_update_profile", lambda: {})(),
        )

    setattr(mod, "_single_node_fast_path_profile", profile)
    return payloads

# --- Core Classes ---

class SparseOptimizer:
    """
    Base class for sparse optimizers.
    It handles updating parameters of modules like DistEmbedding.
    """
    def __init__(self, params: List[Union[DistEmbedding, torch.nn.Module]], lr: float):
        """
        Initializes the optimizer.

        Parameters
        ----------
        params : List[Union[DistEmbedding, torch.nn.Module]]
            A list of modules to be optimized.
        lr : float
            The learning rate.
        """
        self.param_groups = [{"params": params, "lr": lr}]
        self.kv_client = _get_kv_client_if_needed(params)
        self._inflight_handles: List[Tuple[Any, int]] = []
        self._last_update_payloads: List[Dict[str, Any]] = []
        self._last_step_profile: Dict[str, float] = {}
        self.reset_perf_stats()

    def _capture_module_fast_path_profile(self, mod: Any) -> None:
        profile = getattr(mod, "_single_node_fast_path_profile", None)
        if not isinstance(profile, dict):
            return
        normalized_profile = {
            key: float(value) for key, value in profile.items()
        }
        self._last_step_profile = normalized_profile
        self._perf_add("update_owner_exchange_ms", normalized_profile.get("exchange_ms", 0.0))
        self._perf_add("update_trace_merge_ms", normalized_profile.get("owner_aggregate_ms", 0.0))
        self._perf_add("update_local_apply_ms", normalized_profile.get("local_update_ms", 0.0))

    def reset_perf_stats(self) -> None:
        self._perf_stats: Dict[str, float] = {
            "update_trace_merge_ms": 0.0,
            "update_owner_exchange_ms": 0.0,
            "update_local_apply_ms": 0.0,
            "update_async_enqueue_ms": 0.0,
            "update_flush_wait_ms": 0.0,
        }

    def _perf_add(self, key: str, delta_ms: float) -> None:
        self._perf_stats[key] = self._perf_stats.get(key, 0.0) + float(delta_ms)

    def consume_perf_stats(self, reset: bool = True) -> Dict[str, float]:
        stats = dict(self._perf_stats)
        if reset:
            self.reset_perf_stats()
        return stats

    def last_update_payloads(self) -> List[Dict[str, Any]]:
        return [
            {
                **payload,
                "ids": payload["ids"].detach(),
                "grads": payload["grads"].detach(),
            }
            for payload in self._last_update_payloads
        ]

    def step(self):
        """
        Performs a single optimization step.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("The step() method must be implemented by a subclass.")

    def zero_grad(self):
        """
        Clears the traces of all parameter groups.
        """
        for group in self.param_groups:
            for mod in group["params"]:
                if hasattr(mod, 'reset_trace'):
                    mod.reset_trace()
                # else:
                #     if hasattr(mod, 'grad') and mod.grad is not None:
                #         mod.grad.detach_()
                #         mod.grad.zero_()

    def flush(self):
        """Wait for all in-flight async sparse updates."""
        if self.kv_client is None:
            self._inflight_handles.clear()
            return
        for kv_client, handle, payload in self._inflight_handles:
            t_wait_start = perf_counter()
            kv_client.wait(handle)
            self._perf_add("update_flush_wait_ms", (perf_counter() - t_wait_start) * 1e3)
            self._last_update_payloads.append(payload)
        self._inflight_handles.clear()

class SparseSGD(SparseOptimizer):
    def step(self):
        """Performs a single Sparse SGD optimization step."""
        with torch.no_grad():
            self._last_step_profile = {}
            self._last_update_payloads = []
            for group in self.param_groups:
                lr = group["lr"]
                for mod in group["params"]:
                    if isinstance(mod, DistEmbedding):
                        _process_dist_embedding_module(mod, lr)
                    elif hasattr(mod, '_config_names') and hasattr(mod, '_trace'):
                        if _can_use_shared_local_shm_direct_fast_path(mod):
                            self._last_update_payloads.extend(
                                _process_generic_module_with_trace_shared_local_shm_single_table(mod, lr)
                            )
                            self._capture_module_fast_path_profile(mod)
                        elif _can_use_single_node_distributed_fast_path(mod):
                            self._last_update_payloads.extend(
                                _process_generic_module_with_trace_single_node_distributed(mod, lr)
                            )
                            self._capture_module_fast_path_profile(mod)
                        else:
                            t_enqueue_start = perf_counter()
                            self._inflight_handles.extend(
                                _process_generic_module_with_trace(mod, lr, self.kv_client)
                            )
                            self._perf_add("update_async_enqueue_ms", (perf_counter() - t_enqueue_start) * 1e3)
                    else:
                        print(f"Warning: Module type {type(mod).__name__} is not supported by SparseSGD optimizer.")
                    if hasattr(mod, 'reset_trace'):
                        mod.reset_trace()
