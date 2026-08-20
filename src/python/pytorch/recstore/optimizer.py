import json
import math
import os
import time

import torch
from typing import List, Union, Dict, Tuple, Any, Optional
from time import perf_counter
from .single_node_exchange import SparseGradPayload, exchange_sparse_grads

_LOCAL_FAST_PATH_BACKENDS = {"local_shm", "hierkv"}


def _server_sparse_optimizer_config() -> Dict[str, Any]:
    config_path = os.environ.get("RECSTORE_CONFIG")
    if not config_path:
        return {
            "type": "SGD",
            "learning_rate": 0.01,
            "epsilon": 1e-10,
            "beta1": 0.9,
            "beta2": 0.98,
            "weight_decay": 0.0,
        }
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            root_config = json.load(config_file)
        optimizer_config = root_config.get("cache_ps", {}).get("optimizer", {})
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Failed to read sparse optimizer config from {config_path}: {error}"
        ) from error
    if not isinstance(optimizer_config, dict):
        raise RuntimeError("cache_ps.optimizer must be an object")
    optimizer_type = optimizer_config.get("type", "SGD")
    default_epsilon = 1e-8 if optimizer_type == "AdamW" else 1e-10
    return {
        "type": optimizer_type,
        "learning_rate": float(optimizer_config.get("learning_rate", 0.01)),
        "epsilon": float(optimizer_config.get("epsilon", default_epsilon)),
        "beta1": float(optimizer_config.get("beta1", 0.9)),
        "beta2": float(optimizer_config.get("beta2", 0.98)),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
    }


def _validate_server_sparse_optimizer(
    expected_type: str,
    learning_rate: float,
    epsilon: float,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> None:
    actual = _server_sparse_optimizer_config()
    matches = (
        actual["type"] == expected_type
        and math.isclose(
            actual["learning_rate"], float(learning_rate), rel_tol=1e-12, abs_tol=0.0
        )
        and math.isclose(actual["epsilon"], float(epsilon), rel_tol=1e-12, abs_tol=0.0)
    )
    expected_values = {
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
    }
    for key, expected in expected_values.items():
        if expected is not None:
            matches = matches and math.isclose(
                actual[key], float(expected), rel_tol=1e-12, abs_tol=0.0
            )
    if not matches:
        raise RuntimeError(
            "RecStore sparse optimizer mismatch: "
            f"requested type={expected_type}, learning_rate={learning_rate}, "
            f"epsilon={epsilon}; cache_ps config has type={actual['type']}, "
            f"learning_rate={actual['learning_rate']}, epsilon={actual['epsilon']}, "
            f"beta1={actual['beta1']}, beta2={actual['beta2']}, "
            f"weight_decay={actual['weight_decay']}"
        )


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
    current_ps_backend = getattr(module_kv_client, "current_ps_backend", None)
    try:
        direct_sgd = callable(current_ps_backend) and current_ps_backend() == "rdma"
    except Exception:
        direct_sgd = False
    handles = []
    for name, entries in traces_by_name.items():
        if len(entries) == 1:
            all_ids, all_grads = entries[0]
        else:
            all_ids = torch.cat([ids for ids, _ in entries], dim=0)
            all_grads = torch.cat([grads for _, grads in entries], dim=0)
        if direct_sgd:
            update_ids, update_grads = all_ids, all_grads
        else:
            unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)
            update_grads = torch.zeros(
                (len(unique_ids), all_grads.size(1)),
                device=all_grads.device,
                dtype=all_grads.dtype,
            )
            update_grads.index_add_(0, inverse_indices, all_grads)
            update_ids = unique_ids

        # Backend sparse optimizers own learning-rate application for these modules.
        handle = module_kv_client.update_async(name=name, ids=update_ids, grads=update_grads)
        handles.append(
            (
                module_kv_client,
                handle,
                {
                    "module": mod,
                    "name": name,
                    "ids": update_ids.detach(),
                    "grads": update_grads.detach(),
                    "lr": float(lr),
                },
            )
        )
    return handles


def _uses_shared_local_shm_single_table(mod: Any) -> bool:
    if mod.fast_path_mode == "off" or not mod._enable_fusion:
        return False
    if mod._master_config is None:
        return False
    try:
        return bool(mod.kv_client.is_shared_local_shm_table())
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


def _process_generic_module_with_trace_single_node_distributed(
    mod: Any,
    lr: float,
    backend: str,
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
    # ponytail: legacy clients may lack current_ps_backend / activate_shard
    current_backend = (
        module_kv_client.current_ps_backend()
        if hasattr(module_kv_client, "current_ps_backend")
        else None
    )
    if hasattr(module_kv_client, "activate_shard"):
        module_kv_client.activate_shard(rank)
    if current_backend != backend and hasattr(module_kv_client, "set_ps_backend"):
        module_kv_client.set_ps_backend(backend)

    payloads: List[Dict[str, Any]] = []
    traces_by_name = _collect_traces_by_name(mod)
    for name, entries in traces_by_name.items():
        all_ids = torch.cat([ids for ids, _ in entries], dim=0)
        all_grads = torch.cat([grads for _, grads in entries], dim=0)
        unique_ids, summed_grads = _aggregate_ids_and_grads(all_ids, all_grads)
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
        gathered_payloads = exchange_sparse_grads(
            local_payload,
            world_size=world_size,
            backend=exchange_backend,
        )

        owner_ids: List[torch.Tensor] = []
        owner_grads: List[torch.Tensor] = []
        target_device = None
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
    # ponytail: legacy clients may lack current_ps_backend / activate_shard
    current_backend = (
        module_kv_client.current_ps_backend()
        if hasattr(module_kv_client, "current_ps_backend")
        else None
    )
    target_backend = "local_shm"
    if hasattr(module_kv_client, "activate_shard"):
        module_kv_client.activate_shard(rank)
    if current_backend != target_backend and hasattr(module_kv_client, "set_ps_backend"):
        module_kv_client.set_ps_backend(target_backend)

    payloads: List[Dict[str, Any]] = []
    traces_by_name = _collect_traces_by_name(mod)
    for name, entries in traces_by_name.items():
        all_ids = torch.cat([ids for ids, _ in entries], dim=0)
        all_grads = torch.cat([grads for _, grads in entries], dim=0)
        local_unique_ids, local_summed_grads = _aggregate_ids_and_grads(all_ids, all_grads)
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
        self.reset_perf_stats()

    def reset_perf_stats(self) -> None:
        self._perf_stats: Dict[str, float] = {
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
        completed = 0
        try:
            for kv_client, handle, payload in self._inflight_handles:
                t_wait_start = perf_counter()
                kv_client.wait(handle)
                self._perf_add("update_flush_wait_ms", (perf_counter() - t_wait_start) * 1e3)
                self._last_update_payloads.append(payload)
                completed += 1
        finally:
            del self._inflight_handles[:completed]

class SparseSGD(SparseOptimizer):
    def step(self):
        """Aggregates sparse gradients and submits them to the backend optimizer."""
        with torch.no_grad():
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
                        else:
                            fast_path_backend = mod.resolve_fast_path_backend()
                            if fast_path_backend is not None:
                                self._last_update_payloads.extend(
                                    _process_generic_module_with_trace_single_node_distributed(
                                        mod, lr, fast_path_backend
                                    )
                                )
                            else:
                                self._inflight_handles.extend(
                                    _process_generic_module_with_trace(mod, lr, self.kv_client)
                                )
                    else:
                        print(f"Warning: Module type {type(mod).__name__} is not supported by SparseSGD optimizer.")
                    if hasattr(mod, 'reset_trace'):
                        mod.reset_trace()


class SparseRowWiseAdagrad(SparseSGD):
    """Submit raw gradients to a matching server-owned RowWiseAdagrad optimizer."""

    def __init__(
        self,
        params: List[torch.nn.Module],
        lr: float = 0.01,
        eps: float = 1e-10,
    ) -> None:
        if not all(
            hasattr(module, "_config_names") and hasattr(module, "_trace")
            for module in params
        ):
            raise TypeError(
                "SparseRowWiseAdagrad only supports RecStore sparse modules"
            )
        _validate_server_sparse_optimizer("RowWiseAdagrad", lr, eps)
        super().__init__(params, lr)
        self.eps = float(eps)


class SparseAdamW(SparseSGD):
    """Submit raw sparse gradients to the RecStore server-owned AdamW."""

    def __init__(
        self,
        params: List[torch.nn.Module],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not all(
            hasattr(module, "_config_names") and hasattr(module, "_trace")
            for module in params
        ):
            raise TypeError("SparseAdamW only supports RecStore sparse modules")
        beta1, beta2 = betas
        _validate_server_sparse_optimizer(
            "AdamW", lr, eps, beta1=beta1, beta2=beta2, weight_decay=weight_decay
        )
        super().__init__(params, lr)
        self.betas = (float(beta1), float(beta2))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
