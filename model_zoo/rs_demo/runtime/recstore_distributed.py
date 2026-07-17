from __future__ import annotations

import ctypes
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_LOCAL_FAST_PATH_BACKENDS = {"local_shm", "hierkv"}
_NATIVE_DISTRIBUTED_BACKENDS = {"distributed_grpc", "distributed_brpc", "rdma"}
_DEFAULT_LOCAL_SHM_SLOT_BUFFER_BYTES = 8 * 1024 * 1024
_GPU_CACHE_PROFILE_KEYS = (
    "gpu_cache_query_ms",
    "gpu_cache_backend_lookup_ms",
    "gpu_cache_fill_ms",
    "gpu_cache_update_ms",
    "gpu_cache_hit_count",
    "gpu_cache_invalidate_ms",
    "gpu_cache_request_count",
    "gpu_cache_miss_count",
)


@dataclass(frozen=True)
class ShardServer:
    shard: int
    host: str
    port: int


def _load_runtime_config(runtime_dir: Path) -> dict:
    cfg_path = runtime_dir / "recstore_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_client_configs(cfg: dict) -> tuple[dict, dict]:
    distributed = cfg.get("distributed_client")
    cache_ps = cfg.get("cache_ps")
    return (
        distributed if isinstance(distributed, dict) else {},
        cache_ps if isinstance(cache_ps, dict) else {},
    )


def _load_shard_servers(distributed_cfg: dict, cache_cfg: dict, cfg_path: Path) -> list[ShardServer]:
    if "servers" in distributed_cfg:
        raw_servers = distributed_cfg.get("servers") or []
    else:
        raw_servers = cache_cfg.get("servers") or []
    if not raw_servers:
        raise RuntimeError(f"missing shard servers in {cfg_path}")

    servers: list[ShardServer] = [
        ShardServer(
            shard=int(server["shard"]),
            host=str(server["host"]),
            port=int(server["port"]),
        )
        for server in raw_servers
    ]
    servers.sort(key=lambda s: s.shard)
    return servers


def _load_num_shards(distributed_cfg: dict, cache_cfg: dict, servers: list[ShardServer]) -> int:
    if "num_shards" in distributed_cfg:
        return int(distributed_cfg["num_shards"])
    if "num_shards" in cache_cfg:
        return int(cache_cfg["num_shards"])
    return len(servers)


def _load_cache_servers(cache_cfg: dict) -> list[ShardServer]:
    raw_servers = cache_cfg.get("servers") or []
    return [
        ShardServer(
            shard=int(server["shard"]),
            host=str(server["host"]),
            port=int(server["port"]),
        )
        for server in raw_servers
    ]


def _load_hash_method(distributed_cfg: dict) -> str:
    return str(distributed_cfg.get("hash_method", "city_hash"))


def _load_max_keys_per_request(distributed_cfg: dict) -> int:
    try:
        value = int(distributed_cfg.get("max_keys_per_request", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, value)


_CITYHASH_LIB_HANDLE: ctypes.CDLL | None = None
_CITYHASH64_FUNC = None


def _get_cityhash64_func():
    global _CITYHASH_LIB_HANDLE, _CITYHASH64_FUNC
    if _CITYHASH64_FUNC is not None:
        return _CITYHASH64_FUNC
    lib_path = Path(__file__).resolve().parents[3] / "third_party/cityhash/src/.libs/libcityhash.so.0.0.0"
    try:
        _CITYHASH_LIB_HANDLE = ctypes.CDLL(str(lib_path))
    except OSError as exc:
        raise RuntimeError(f"failed to load cityhash library: {lib_path}") from exc
    try:
        func = _CITYHASH_LIB_HANDLE._Z10CityHash64PKcm
    except AttributeError as exc:
        raise RuntimeError(f"missing CityHash64 symbol in library: {lib_path}") from exc
    func.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    func.restype = ctypes.c_uint64
    _CITYHASH64_FUNC = func
    return _CITYHASH64_FUNC


def _city_hash64_of_uint64(key: int) -> int:
    raw = struct.pack("<Q", int(key) & 0xFFFFFFFFFFFFFFFF)
    city_hash64 = _get_cityhash64_func()
    return int(city_hash64(raw, len(raw)))


class ShardedRecstoreClient:
    def __init__(self, client, runtime_dir: Path) -> None:
        cfg = _load_runtime_config(runtime_dir)
        distributed_cfg, cache_cfg = _load_client_configs(cfg)
        self._client = client
        self._cache_ps_type = str(cache_cfg.get("ps_type", "")).upper()
        local_shm_cfg = cfg.get("local_shm")
        self._local_shm_slot_buffer_bytes = _DEFAULT_LOCAL_SHM_SLOT_BUFFER_BYTES
        if isinstance(local_shm_cfg, dict):
            self._local_shm_slot_buffer_bytes = int(
                local_shm_cfg.get(
                    "slot_buffer_bytes",
                    _DEFAULT_LOCAL_SHM_SLOT_BUFFER_BYTES,
                )
            )
        self._cache_servers = _load_cache_servers(cache_cfg)
        self._cache_num_shards = _load_num_shards({}, cache_cfg, self._cache_servers)
        self._servers = _load_shard_servers(
            distributed_cfg, cache_cfg, runtime_dir / "recstore_config.json"
        )
        self._servers_by_shard = {server.shard: server for server in self._servers}
        self._num_shards = _load_num_shards(distributed_cfg, cache_cfg, self._servers)
        self._hash_method = _load_hash_method(distributed_cfg)
        self._max_keys_per_request = _load_max_keys_per_request(distributed_cfg)
        self._active_shard: int | None = None
        self._next_prefetch_id = 1
        self._prefetch_contexts: dict[int, tuple[int, list[tuple[int, torch.Tensor, Any]]]] = {}
        self._kv_prefetch_next_id = 1
        self._kv_prefetch_contexts: dict[int, tuple[int, list[tuple[int, torch.Tensor, Any]]]] = {}
        self._kv_prefetch_ready_results: dict[int, torch.Tensor] = {}
        self._tensor_meta: Dict[str, Dict[str, Any]] = {}
        self._next_async_handle = 1
        self._pending_async_ops: Dict[int, tuple[str, torch.Tensor, torch.Tensor]] = {}
        self._gpu_cache_table_name: str | None = None
        self._gpu_cache_enabled: bool = False
        self._gpu_cache_clear_count = 0
        self._clear_gpu_cache_after_cpu_update = True
        self._prefetch_table_name: str | None = None
        self._native_distributed_backend = self._detect_native_distributed_backend()

    def register_tensor_meta(
        self,
        name: str,
        shape: tuple[int, int],
        dtype: torch.dtype,
        base_offset: int = 0,
    ) -> None:
        if name in self._tensor_meta:
            return
        num_embeddings, embedding_dim = shape
        self._tensor_meta[name] = {
            "shape": (int(num_embeddings), int(embedding_dim)),
            "dtype": dtype,
            "base_offset": int(base_offset),
        }

    def _shard_for_key(self, key: int) -> int:
        if self._hash_method == "simple_mod":
            return int(key) % self._num_shards
        return _city_hash64_of_uint64(int(key)) % self._num_shards

    def _activate_shard(self, shard: int) -> None:
        shard = int(shard)
        if self._active_shard == shard:
            return
        if self._uses_native_distributed_backend():
            self._active_shard = shard
            return
        if self.is_shared_local_shm_table():
            self._active_shard = shard
            return
        current_backend = None
        if hasattr(self._client, "ops") and hasattr(self._client.ops, "current_ps_backend"):
            current_backend = str(self._client.ops.current_ps_backend())
        server = self._servers_by_shard.get(shard)
        if server is None:
            raise RuntimeError(f"no server configured for shard {shard}")
        if current_backend not in _LOCAL_FAST_PATH_BACKENDS:
            self._client.ops.set_ps_config(server.host, server.port)
        self._active_shard = shard

    def activate_shard(self, shard: int) -> None:
        self._activate_shard(shard)

    def _group_indices(self, keys: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        shard_to_indices: dict[int, list[int]] = {}
        shard_order: list[int] = []
        for idx, key in enumerate(keys.tolist()):
            shard = self._shard_for_key(int(key))
            if shard not in shard_to_indices:
                shard_to_indices[shard] = []
                shard_order.append(shard)
            shard_to_indices[shard].append(idx)
        return [
            (shard, torch.tensor(indices, dtype=torch.long))
            for shard, indices in ((shard, shard_to_indices[shard]) for shard in shard_order)
        ]

    def _normalize_ids(self, ids: torch.Tensor, *, keep_device: bool = False) -> torch.Tensor:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")
        normalized_ids = ids.to(dtype=torch.int64)
        if not normalized_ids.is_contiguous():
            normalized_ids = normalized_ids.contiguous()
        if not keep_device and normalized_ids.device.type != "cpu":
            normalized_ids = normalized_ids.to("cpu")
        return normalized_ids

    def _normalize_grads(self, grads: torch.Tensor, *, keep_device: bool = False) -> torch.Tensor:
        if not isinstance(grads, torch.Tensor):
            raise TypeError("grads must be a torch.Tensor")
        normalized_grads = grads.to(dtype=torch.float32)
        if not normalized_grads.is_contiguous():
            normalized_grads = normalized_grads.contiguous()
        if not keep_device and normalized_grads.device.type != "cpu":
            normalized_grads = normalized_grads.to("cpu")
        return normalized_grads

    def _normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        normalized_values = values.to(dtype=torch.float32)
        if not normalized_values.is_contiguous():
            normalized_values = normalized_values.contiguous()
        if normalized_values.device.type != "cpu":
            normalized_values = normalized_values.to("cpu")
        return normalized_values

    def _require_active_shard(self, api_name: str) -> None:
        if self._active_shard is None:
            raise RuntimeError(f"{api_name} requires an active shard to be selected first.")

    def _detect_native_distributed_backend(self) -> bool:
        ops = getattr(self._client, "ops", None)
        getter = getattr(ops, "current_ps_backend", None)
        if not callable(getter):
            return False
        try:
            return str(getter()).lower() in _NATIVE_DISTRIBUTED_BACKENDS
        except Exception:
            return False

    def _uses_native_distributed_backend(self) -> bool:
        return self._native_distributed_backend

    def _can_issue_shard_prefetch_early(self) -> bool:
        return self._uses_native_distributed_backend() or self._num_shards <= 1

    def _require_local_shm_backend(self, api_name: str) -> None:
        if not hasattr(self._client, "ops") or not hasattr(self._client.ops, "current_ps_backend"):
            raise RuntimeError(
                f"{api_name} requires a RecStore ops library exposing current_ps_backend()."
            )
        backend = self._client.ops.current_ps_backend()
        if backend not in _LOCAL_FAST_PATH_BACKENDS:
            raise RuntimeError(
                f"{api_name} requires local_shm or hierkv backend, but current backend is {backend}."
            )

    def _group_ids_by_shard(
        self,
        keys: torch.Tensor,
        *,
        keep_key_device: bool = False,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        ids_cpu = self._normalize_ids(keys)
        shard_to_indices: dict[int, list[int]] = {}
        shard_order: list[int] = []
        for idx, key in enumerate(ids_cpu.tolist()):
            shard = self._shard_for_key(int(key))
            if shard not in shard_to_indices:
                shard_to_indices[shard] = []
                shard_order.append(shard)
            shard_to_indices[shard].append(idx)
        grouped: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for shard in shard_order:
            indices = shard_to_indices[shard]
            index_tensor = torch.tensor(indices, dtype=torch.long)
            key_source = keys if keep_key_device else ids_cpu
            if keep_key_device and key_source.device.type != "cpu":
                index_for_source = index_tensor.to(key_source.device)
            else:
                index_for_source = index_tensor
            shard_keys = key_source.index_select(0, index_for_source)
            if not shard_keys.is_contiguous():
                shard_keys = shard_keys.contiguous()
            grouped.append((shard, index_tensor, shard_keys))
        return grouped

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        if self._uses_native_distributed_backend():
            return bool(
                self._client.init_embedding_table(
                    table_name, num_embeddings, embedding_dim
                )
            )
        ok = True
        for server in self._servers:
            self._activate_shard(server.shard)
            ok = self._client.init_embedding_table(table_name, num_embeddings, embedding_dim) and ok
        return ok

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        if self._uses_native_distributed_backend():
            self._client.emb_write(self._normalize_ids(keys), self._normalize_values(values))
            return
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_vals = values.index_select(0, index_tensor).contiguous()
            self._client.emb_write(shard_keys, shard_vals)

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        if keys.numel() == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32)
        if self._uses_native_distributed_backend():
            return self._client.emb_read(self._normalize_ids(keys), embedding_dim)
        out = torch.empty((keys.shape[0], embedding_dim), dtype=torch.float32)
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_values = self._client.emb_read(shard_keys, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        return out

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        if self._uses_native_distributed_backend():
            return int(self._client.emb_prefetch(self._normalize_ids(keys)))
        req_id = self._next_prefetch_id
        self._next_prefetch_id += 1
        if keys.numel() == 0:
            self._prefetch_contexts[req_id] = (0, [])
            return req_id

        shard_requests: list[tuple[int, torch.Tensor, Any]] = []
        for shard, index_tensor in self._group_indices(keys):
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            if self._can_issue_shard_prefetch_early():
                self._activate_shard(shard)
                shard_prefetch_id = int(self._client.emb_prefetch(shard_keys))
                shard_requests.append((shard, index_tensor, shard_prefetch_id))
            else:
                shard_requests.append((shard, index_tensor, shard_keys))
        self._prefetch_contexts[req_id] = (int(keys.shape[0]), shard_requests)
        return req_id

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        if self._uses_native_distributed_backend():
            return self._client.emb_wait_result(int(prefetch_id), embedding_dim)
        context = self._prefetch_contexts.pop(int(prefetch_id), None)
        if context is None:
            raise RuntimeError(f"unknown prefetch_id: {prefetch_id}")
        total_rows, shard_requests = context
        if total_rows == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32)
        out = torch.empty((total_rows, embedding_dim), dtype=torch.float32)
        for shard, index_tensor, request in shard_requests:
            self._activate_shard(shard)
            if isinstance(request, torch.Tensor):
                shard_prefetch_id = int(self._client.emb_prefetch(request))
            else:
                shard_prefetch_id = int(request)
            shard_values = self._client.emb_wait_result(shard_prefetch_id, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        return out

    def emb_read_prefetch(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        prefetch_id = self.emb_prefetch(keys)
        return self.emb_wait_result(prefetch_id, embedding_dim)

    def current_ps_backend(self) -> str:
        if not hasattr(self._client, "ops") or not hasattr(self._client.ops, "current_ps_backend"):
            raise RuntimeError(
                "current_ps_backend requires a RecStore ops library exposing current_ps_backend()."
            )
        return str(self._client.ops.current_ps_backend())

    def activate_shard(self, shard: int) -> None:
        self._activate_shard(int(shard))

    def set_ps_backend(self, backend: str) -> None:
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string")
        if not hasattr(self._client, "ops") or not hasattr(self._client.ops, "set_ps_backend"):
            raise RuntimeError(
                "set_ps_backend requires a RecStore ops library exposing set_ps_backend()."
            )
        self._client.ops.set_ps_backend(backend)
        self._native_distributed_backend = self._detect_native_distributed_backend()

    def enable_gpu_cache(self, capacity: int, embedding_dim: int) -> bool:
        enable = getattr(self._client, "enable_gpu_cache", None)
        if not callable(enable):
            ops = getattr(self._client, "ops", None)
            enable = getattr(ops, "enable_gpu_cache", None)
        if not callable(enable):
            raise RuntimeError(
                "enable_gpu_cache requires a RecStore client or ops library exposing "
                "enable_gpu_cache()."
        )
        enabled = bool(enable(int(capacity), int(embedding_dim)))
        self._gpu_cache_enabled = enabled
        return enabled

    def is_gpu_cache_enabled(self) -> bool:
        return bool(self._gpu_cache_enabled)

    def set_gpu_cache_lookup_bypass_enabled(self, enabled: bool) -> None:
        setter = getattr(self._client, "set_gpu_cache_lookup_bypass_enabled", None)
        if not callable(setter):
            ops = getattr(self._client, "ops", None)
            setter = getattr(ops, "set_gpu_cache_lookup_bypass_enabled", None)
        if not callable(setter):
            raise RuntimeError(
                "set_gpu_cache_lookup_bypass_enabled requires a RecStore client or "
                "ops library exposing set_gpu_cache_lookup_bypass_enabled()."
            )
        setter(bool(enabled))

    def is_gpu_cache_lookup_bypass_enabled(self) -> bool:
        getter = getattr(self._client, "is_gpu_cache_lookup_bypass_enabled", None)
        if not callable(getter):
            ops = getattr(self._client, "ops", None)
            getter = getattr(ops, "is_gpu_cache_lookup_bypass_enabled", None)
        if not callable(getter):
            raise RuntimeError(
                "is_gpu_cache_lookup_bypass_enabled requires a RecStore client or "
                "ops library exposing is_gpu_cache_lookup_bypass_enabled()."
            )
        return bool(getter())

    def is_gpu_cache_lookup_bypassed(self) -> bool:
        getter = getattr(self._client, "is_gpu_cache_lookup_bypassed", None)
        if not callable(getter):
            ops = getattr(self._client, "ops", None)
            getter = getattr(ops, "is_gpu_cache_lookup_bypassed", None)
        if not callable(getter):
            raise RuntimeError(
                "is_gpu_cache_lookup_bypassed requires a RecStore client or ops "
                "library exposing is_gpu_cache_lookup_bypassed()."
            )
        return bool(getter())

    def reset_gpu_cache_bypass_state(self) -> None:
        reset = getattr(self._client, "reset_gpu_cache_bypass_state", None)
        if not callable(reset):
            ops = getattr(self._client, "ops", None)
            reset = getattr(ops, "reset_gpu_cache_bypass_state", None)
        if not callable(reset):
            raise RuntimeError(
                "reset_gpu_cache_bypass_state requires a RecStore client or ops "
                "library exposing reset_gpu_cache_bypass_state()."
            )
        reset()

    def prefill_gpu_cache(
        self,
        name: str,
        ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if not self._gpu_cache_enabled:
            raise RuntimeError("prefill_gpu_cache requires GPU cache to be enabled")
        prefill = getattr(self._client, "prefill_gpu_cache", None)
        if callable(prefill):
            prefill(name, ids, values)
            return
        ops = getattr(self._client, "ops", None)
        prefill = getattr(ops, "prefill_gpu_cache", None)
        if not callable(prefill):
            raise RuntimeError(
                "prefill_gpu_cache requires a RecStore client or ops library "
                "exposing prefill_gpu_cache()."
            )
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        normalized_values = self._normalize_grads(values, keep_device=True)
        if (
            normalized_ids.numel() > 0
            and normalized_ids.device.type == "cpu"
            and normalized_values.device.type == "cpu"
            and torch.cuda.is_available()
        ):
            cache_device = torch.device("cuda", torch.cuda.current_device())
            normalized_ids = normalized_ids.to(cache_device)
            normalized_values = normalized_values.to(cache_device)
        prefill(normalized_ids, normalized_values)

    def invalidate_gpu_cache(self, name: str, ids: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        if not self._gpu_cache_enabled:
            raise RuntimeError("invalidate_gpu_cache requires GPU cache to be enabled")
        invalidator = getattr(self._client, "invalidate_gpu_cache", None)
        if callable(invalidator):
            invalidator(name, ids)
            return
        ops = getattr(self._client, "ops", None)
        invalidator = getattr(ops, "invalidate_gpu_cache", None)
        if not callable(invalidator):
            raise RuntimeError(
                "invalidate_gpu_cache requires a RecStore client or ops library "
                "exposing invalidate_gpu_cache()."
            )
        self._ensure_gpu_cache_table(name)
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        if normalized_ids.numel() > 0 and normalized_ids.device.type == "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError("invalidate_gpu_cache requires CUDA ids")
            normalized_ids = normalized_ids.to(torch.device("cuda", torch.cuda.current_device()))
        invalidator(normalized_ids)

    def apply_sgd_update_gpu_cache(
        self,
        name: str,
        ids: torch.Tensor,
        grads: torch.Tensor,
        *,
        learning_rate: float,
    ) -> bool:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        if not self._gpu_cache_enabled:
            raise RuntimeError("apply_sgd_update_gpu_cache requires GPU cache to be enabled")
        self._ensure_gpu_cache_table(name)
        ops = getattr(self._client, "ops", None)
        updater = getattr(ops, "apply_sgd_update_gpu_cache", None)
        if not callable(updater):
            updater = getattr(self._client, "apply_sgd_update_gpu_cache", None)
            if callable(updater):
                return bool(
                    updater(
                        name,
                        ids,
                        grads,
                        learning_rate=float(learning_rate),
                    )
                )
            raise RuntimeError(
                "apply_sgd_update_gpu_cache requires a RecStore client or ops library "
                "exposing apply_sgd_update_gpu_cache()."
            )
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        normalized_grads = self._normalize_grads(grads, keep_device=True)
        if normalized_grads.dim() != 2:
            raise ValueError("grads must be a 2-dimensional tensor")
        if normalized_ids.size(0) != normalized_grads.size(0):
            raise ValueError("ids and grads must have the same number of rows")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        if normalized_grads.size(1) != embedding_dim:
            raise ValueError(
                f"grads second dimension must match embedding dim {embedding_dim} for tensor '{name}'"
            )
        if (
            normalized_ids.numel() > 0
            and normalized_ids.device.type == "cpu"
            and normalized_grads.device.type == "cpu"
            and torch.cuda.is_available()
        ):
            cache_device = torch.device("cuda", torch.cuda.current_device())
            normalized_grads = normalized_grads.to(cache_device)
        return bool(updater(normalized_ids, normalized_grads, float(learning_rate)))

    def gpu_cache_lookup_flat(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Client-side GPU cache lookup that works with any backend."""
        if not self._gpu_cache_enabled:
            raise RuntimeError("gpu_cache_lookup_flat requires GPU cache to be enabled")
        fn = getattr(self._client, "gpu_cache_lookup_flat", None)
        if not callable(fn):
            raise RuntimeError(
                "gpu_cache_lookup_flat requires a RecStore client exposing "
                "gpu_cache_lookup_flat()."
            )
        normalized = self._normalize_ids(keys, keep_device=True)
        if not normalized.is_contiguous():
            normalized = normalized.contiguous()
        return fn(normalized, int(embedding_dim))

    def query_gpu_cache(self, keys: torch.Tensor, embedding_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        fn = getattr(self._client, "query_gpu_cache", None)
        if not callable(fn):
            raise RuntimeError("query_gpu_cache requires client support.")
        return fn(self._normalize_ids(keys, keep_device=True), int(embedding_dim))

    def update_gpu_cache(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        fn = getattr(self._client, "update_gpu_cache", None)
        if not callable(fn):
            raise RuntimeError("update_gpu_cache requires client support.")
        fn(self._normalize_ids(ids, keep_device=True), self._normalize_grads(values, keep_device=True))

    def get_last_gpu_cache_profile(self) -> dict[str, float]:
        getter = getattr(self._client, "get_last_gpu_cache_profile", None)
        if callable(getter):
            profile = getter()
            return profile if isinstance(profile, dict) else {}
        ops = getattr(self._client, "ops", None)
        getter = getattr(ops, "get_last_gpu_cache_profile", None)
        if not callable(getter):
            return {}
        values = getter()
        if not isinstance(values, (list, tuple)) or len(values) < 5:
            return {}
        profile = {}
        for index, key in enumerate(_GPU_CACHE_PROFILE_KEYS):
            profile[key] = float(values[index]) if index < len(values) else 0.0
        return profile

    def get_gpu_cache_clear_count(self) -> int:
        return int(self._gpu_cache_clear_count)

    def set_clear_gpu_cache_after_cpu_update(self, enabled: bool) -> None:
        self._clear_gpu_cache_after_cpu_update = bool(enabled)

    def _clear_gpu_cache_if_available(self) -> None:
        clear = getattr(self._client, "clear_gpu_cache", None)
        if not callable(clear):
            ops = getattr(self._client, "ops", None)
            clear = getattr(ops, "clear_gpu_cache", None)
        if callable(clear):
            clear()
            self._gpu_cache_clear_count += 1
        self._gpu_cache_table_name = None

    def _max_emb_write_rows(self, embedding_dim: int) -> int:
        if self._cache_ps_type != "LOCAL_SHM":
            return self._max_keys_per_request
        bytes_per_row = 8 + int(embedding_dim) * 4
        local_limit = (
            max(1, int(self._local_shm_slot_buffer_bytes) // bytes_per_row)
            if bytes_per_row > 0
            else 0
        )
        if self._max_keys_per_request > 0:
            return min(local_limit, self._max_keys_per_request)
        return local_limit

    def _emb_write_chunked_if_needed(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        embedding_dim: int,
    ) -> None:
        max_rows = self._max_emb_write_rows(embedding_dim)
        if max_rows <= 0 or keys.numel() <= max_rows:
            self._client.emb_write(keys, values)
            return
        for start in range(0, int(keys.numel()), max_rows):
            end = min(start + max_rows, int(keys.numel()))
            self._client.emb_write(keys[start:end], values[start:end])

    def _ensure_gpu_cache_table(self, name: str) -> None:
        if self._gpu_cache_table_name is None:
            self._gpu_cache_table_name = name
            return
        if self._gpu_cache_table_name != name:
            self._clear_gpu_cache_if_available()
            self._gpu_cache_table_name = name

    def set_prefetch_table_name(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("prefetch table name must be a non-empty string")
        self._prefetch_table_name = name

    def is_shared_local_shm_table(self) -> bool:
        if self._cache_ps_type != "LOCAL_SHM":
            return False
        if self._cache_num_shards != 1:
            return False
        if len(self._cache_servers) != 1:
            return False
        return True

    def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._require_active_shard("local_lookup_flat")
        self._require_local_shm_backend("local_lookup_flat")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        self._ensure_gpu_cache_table(name)
        return self._client.ops.local_lookup_flat(normalized_ids, embedding_dim)

    def warmup_local_lookup_flat_cuda_region(self) -> bool:
        self._require_active_shard("warmup_local_lookup_flat_cuda_region")
        warmup = getattr(self._client.ops, "warmup_local_lookup_flat_cuda_region", None)
        if not callable(warmup):
            return False
        return bool(warmup())

    def get_last_local_shm_lookup_profile(self) -> dict[str, float]:
        getter = getattr(self._client.ops, "get_last_local_lookup_flat_profile", None)
        if not callable(getter):
            return {}
        values = getter()
        if not isinstance(values, (list, tuple)) or len(values) < 7:
            return {}
        return {
            "lookup_cpp_total_ms": float(values[0]),
            "lookup_keys_stage_ms": float(values[1]),
            "lookup_submit_ms": float(values[2]),
            "lookup_wait_ms": float(values[3]),
            "lookup_payload_pin_ms": float(values[4]),
            "lookup_fallback_copy_ms": float(values[5]),
            "lookup_values_h2d_enqueue_ms": float(values[6]),
        }

    def local_update_flat(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._require_active_shard("local_update_flat")
        self._require_local_shm_backend("local_update_flat")
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        normalized_grads = self._normalize_grads(grads, keep_device=True)
        self._ensure_gpu_cache_table(name)
        if normalized_grads.dim() != 2:
            raise ValueError("grads must be a 2-dimensional tensor")
        if normalized_ids.size(0) != normalized_grads.size(0):
            raise ValueError("ids and grads must have the same number of rows")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        if normalized_grads.size(1) != embedding_dim:
            raise ValueError(
                f"grads second dimension must match embedding dim {embedding_dim} for tensor '{name}'"
            )
        self._client.ops.local_update_flat(name, normalized_ids, normalized_grads)
        if normalized_ids.device.type == "cpu":
            self._clear_gpu_cache_if_available()

    def get_last_local_shm_update_profile(self) -> dict[str, float]:
        getter = getattr(self._client.ops, "get_last_local_update_flat_profile", None)
        if not callable(getter):
            return {}
        values = getter()
        if not isinstance(values, (list, tuple)) or len(values) < 4:
            return {}
        return {
            "local_update_cpp_total_ms": float(values[0]),
            "local_update_keys_stage_ms": float(values[1]),
            "local_update_grads_stage_ms": float(values[2]),
            "local_update_shm_call_ms": float(values[3]),
            "local_update_backend_call_ms": float(values[3]),
            "local_update_stage_wait_ms": float(values[4]) if len(values) > 4 else 0.0,
        }

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        if self._uses_native_distributed_backend():
            self._client.emb_update_table(
                table_name,
                self._normalize_ids(keys),
                self._normalize_grads(grads),
            )
            return
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_grads = grads.index_select(0, index_tensor).contiguous()
            self._client.emb_update_table(table_name, shard_keys, shard_grads)

    def emb_write_values(self, name: str, ids: torch.Tensor, values: torch.Tensor) -> None:
        """Write (set) embedding values directly to the PS for a subset of
        keys with per-key GPU cache invalidation (no full cache clear).

        Used by the BagPipe eviction writeback path.  Routes through the
        native distributed backend (single emb_write_values call) or, for
        sharded backends, groups keys by shard and writes each shard.
        """
        if ids.numel() == 0:
            return
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor {name} has not been initialized.")
        ids_n = self._normalize_ids(ids, keep_device=True)
        vals_n = self._normalize_grads(values, keep_device=True)
        if ids_n.size(0) != vals_n.size(0):
            raise RuntimeError("ids and values must have the same number of rows for emb_write_values")
        if self._uses_native_distributed_backend():
            self._client.emb_write_values(name, ids_n, vals_n)
            return
        for shard, index_tensor, shard_keys in self._group_ids_by_shard(
            ids_n, keep_key_device=True
        ):
            self._activate_shard(shard)
            idx = index_tensor
            if vals_n.device != idx.device:
                idx = idx.to(vals_n.device)
            shard_vals = vals_n.index_select(0, idx).contiguous()
            self._client.emb_write_values(name, shard_keys, shard_vals)

    def update(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> None:
        """Push gradients to the PS and invalidate ONLY these keys from the
        GPU cache (not a full cache clear).

        This routes through emb_update_table with device-preserved keys so
        the C++ cache maintenance takes the specific-key invalidation path
        (InvalidateGpuCache) instead of clearing the whole cache.  The BagPipe
        controller relies on this for sync_now/sync_later grad flush so that
        non-evicted cache entries survive.
        """
        if ids.numel() == 0:
            return
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor {name} has not been initialized.")
        ids_n = self._normalize_ids(ids, keep_device=True)
        grads_n = self._normalize_grads(grads, keep_device=True)
        if ids_n.size(0) != grads_n.size(0):
            raise RuntimeError("ids and grads must have the same number of rows for update")
        if self._uses_native_distributed_backend():
            self._client.emb_update_table(name, ids_n, grads_n)
            return
        for shard, index_tensor, shard_keys in self._group_ids_by_shard(
            ids_n, keep_key_device=True
        ):
            self._activate_shard(shard)
            idx = index_tensor
            if grads_n.device != idx.device:
                idx = idx.to(grads_n.device)
            shard_grads = grads_n.index_select(0, idx).contiguous()
            self._client.emb_update_table(name, shard_keys, shard_grads)

    def init_data(
        self,
        name: str,
        shape: tuple[int, int],
        dtype: torch.dtype,
        part_policy: Any = None,
        init_func: Any | None = None,
        is_gdata: bool = True,
        base_offset: int = 0,
    ) -> None:
        if name in self._tensor_meta:
            return
        num_embeddings, embedding_dim = shape
        self.register_tensor_meta(name=name, shape=shape, dtype=dtype, base_offset=base_offset)
        self.init_embedding_table(name, num_embeddings, embedding_dim)

        if init_func:
            initial_data = init_func(shape, dtype)
        else:
            initial_data = torch.zeros(shape, dtype=dtype)
        if not isinstance(initial_data, torch.Tensor):
            initial_data = torch.tensor(initial_data, dtype=dtype)
        initial_data = initial_data.to(dtype=dtype)
        if initial_data.device.type != "cpu":
            initial_data = initial_data.to("cpu")
        if not initial_data.is_contiguous():
            initial_data = initial_data.contiguous()

        keys = torch.arange(num_embeddings, dtype=torch.int64)
        if base_offset != 0:
            keys = keys + int(base_offset)

        if self._uses_native_distributed_backend():
            self._emb_write_chunked_if_needed(keys, initial_data, embedding_dim)
            return

        for shard, index_tensor, shard_keys in self._group_ids_by_shard(keys):
            self._activate_shard(shard)
            shard_values = initial_data.index_select(0, index_tensor).contiguous()
            self._emb_write_chunked_if_needed(shard_keys, shard_values, embedding_dim)

    def pull(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor {name} has not been initialized.")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        return self.emb_read(ids, embedding_dim)

    def pull_with_gpu_cache(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor {name} has not been initialized.")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        self._ensure_gpu_cache_table(name)
        if self._uses_native_distributed_backend():
            return self._client.emb_read(normalized_ids, embedding_dim)
        if self._num_shards != 1:
            raise RuntimeError(
                "pull_with_gpu_cache currently supports single-shard distributed "
                "wrappers only; multi-shard GPU scatter is not implemented"
            )
        server = self._servers[0]
        self._activate_shard(server.shard)
        return self._client.emb_read(normalized_ids, embedding_dim)

    def prefetch(self, ids: torch.Tensor) -> int:
        normalized_ids = self._normalize_ids(ids)
        if self._uses_native_distributed_backend():
            return int(self._client.emb_prefetch(normalized_ids))
        req_id = self._kv_prefetch_next_id
        self._kv_prefetch_next_id += 1
        if normalized_ids.numel() == 0:
            self._kv_prefetch_contexts[req_id] = (0, [])
            return req_id
        prefetch_table_name = self._prefetch_table_name or self._gpu_cache_table_name
        if self.is_shared_local_shm_table() and prefetch_table_name is not None:
            self._activate_shard(0)
            self._kv_prefetch_ready_results[req_id] = self.local_lookup_flat(
                prefetch_table_name,
                normalized_ids,
            )
            self._kv_prefetch_contexts[req_id] = (int(normalized_ids.numel()), [])
            return req_id
        shard_requests: list[tuple[int, torch.Tensor, Any]] = []
        for shard, index_tensor, shard_keys in self._group_ids_by_shard(normalized_ids):
            if self._can_issue_shard_prefetch_early():
                self._activate_shard(shard)
                shard_prefetch_id = int(self._client.emb_prefetch(shard_keys))
                shard_requests.append((shard, index_tensor, shard_prefetch_id))
            else:
                shard_requests.append((shard, index_tensor, shard_keys))
        self._kv_prefetch_contexts[req_id] = (int(normalized_ids.numel()), shard_requests)
        return req_id

    def wait_and_get(
        self,
        prefetch_id: int,
        embedding_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        if self._uses_native_distributed_backend():
            out = self._client.emb_wait_result(int(prefetch_id), embedding_dim)
            if device.type == "cuda":
                out = out.to(device)
            return out
        context = self._kv_prefetch_contexts.pop(int(prefetch_id), None)
        if context is None:
            raise RuntimeError(f"unknown prefetch_id: {prefetch_id}")
        total_rows, shard_requests = context
        ready_result = self._kv_prefetch_ready_results.pop(int(prefetch_id), None)
        if ready_result is not None:
            out = ready_result
            if out.size(1) != int(embedding_dim):
                raise RuntimeError(
                    f"prefetch result embedding dim {out.size(1)} does not match requested {embedding_dim}"
                )
            if device.type == "cuda" and out.device.type != "cuda":
                out = out.to(device)
            return out
        if total_rows == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32, device=device)
        out = torch.empty((total_rows, embedding_dim), dtype=torch.float32)
        for shard, index_tensor, request in shard_requests:
            self._activate_shard(shard)
            if isinstance(request, torch.Tensor):
                shard_prefetch_id = int(self._client.emb_prefetch(request))
            else:
                shard_prefetch_id = int(request)
            shard_values = self._client.emb_wait_result(shard_prefetch_id, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        if device.type == "cuda":
            out = out.to(device)
        return out

    def update_async(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> int:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor {name} has not been initialized.")
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        normalized_grads = self._normalize_grads(grads, keep_device=True)
        if normalized_ids.size(0) != normalized_grads.size(0):
            raise RuntimeError("ids and grads must have the same number of rows for update_async")
        handle = self._next_async_handle
        self._next_async_handle += 1
        self._pending_async_ops[handle] = (name, normalized_ids, normalized_grads)
        return handle

    def wait(self, handle: int) -> None:
        pending = self._pending_async_ops.pop(int(handle), None)
        if pending is None:
            return
        name, ids, grads = pending
        if ids.numel() == 0:
            return
        updated_on_cpu = ids.device.type == "cpu"
        if self._uses_native_distributed_backend():
            self._client.emb_update_table(
                name,
                self._normalize_ids(ids),
                self._normalize_grads(grads),
            )
            if updated_on_cpu and self._clear_gpu_cache_after_cpu_update:
                self._clear_gpu_cache_if_available()
            return
        for shard, index_tensor, shard_keys in self._group_ids_by_shard(
            ids,
            keep_key_device=True,
        ):
            if grads.device.type != "cpu":
                grad_index = index_tensor.to(grads.device)
            else:
                grad_index = index_tensor
            shard_grads = grads.index_select(0, grad_index).contiguous()
            self._activate_shard(shard)
            self._client.emb_update_table(name, shard_keys, shard_grads)
        if updated_on_cpu and self._clear_gpu_cache_after_cpu_update:
            self._clear_gpu_cache_if_available()
