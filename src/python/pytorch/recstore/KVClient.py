import torch
import os
import time
import ctypes
from typing import Optional, Tuple, List, Dict, Any, Callable

_KEY_TAG_SHIFT = 56
_LOCAL_FAST_PATH_BACKENDS = {"local_shm", "hierkv"}

def get_reporter():
    if not hasattr(get_reporter, 'lib'):
        script_dir = os.path.dirname(__file__)
        lib_path = os.path.abspath(os.path.join(script_dir, '../../../../build/lib/libreport.so'))
        if os.path.exists(lib_path):
            lib = ctypes.CDLL(lib_path)
            lib.report.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
            lib.report.restype = ctypes.c_bool
            get_reporter.lib = lib
        else:
            get_reporter.lib = None
    return get_reporter.lib

def report_metric(table: str, uid: str, metric: str, value: float) -> bool:
    lib = get_reporter()
    if lib:
        return lib.report(table.encode('utf-8'), uid.encode('utf-8'), metric.encode('utf-8'), float(value))
    return False

class RecStoreClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RecStoreClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, role: str = "default"):
        if self._initialized:
            return

        self.ops = torch.ops.recstore_ops

        self._part_policy = {}
        
        self._tensor_meta = {}
        self._full_data_shape = {}
        self._data_name_list = set()
        self._gdata_name_list = set()
        self._role = role
        self._next_async_handle = 1
        self._pending_async_ops = {}
        self._gpu_cache_table_name: Optional[str] = None
        self._gpu_cache_enabled = False
        self._gpu_cache_clear_count = 0
        self._clear_gpu_cache_after_cpu_update = True
        self._prefetch_table_name: Optional[str] = None
        self._next_table_id = 0
        self._initialized = True
    @property
    def role(self) -> str:
        """Get client role"""
        return self._role

    @property
    def client_id(self) -> int:
        """Get client ID"""
        # This is a mock value as there's no RPC-based client ID.
        return 0

    @property
    def machine_id(self) -> int:
        """Get machine ID"""
        # This is a mock value as there's no distributed setup.
        return 0

    @property
    def part_policy(self):
        """Get part policy"""
        return self._part_policy
        
    def num_servers(self) -> int:
        """Get the number of servers"""
        # In our mock setup, this is always 1.
        return 1

    def barrier(self):
        """Barrier for all client nodes.

        This API will be blocked untill all the clients invoke this API.
        """
        # Not applicable in a non-distributed, ops-based setup.
        print("Warning: barrier() called but has no effect in ops-based implementation.")
        pass

    def register_push_handler(self, name: str, func: Callable):
        """Register UDF push function."""
        raise NotImplementedError("register_push_handler is not implemented for the ops-based client.")

    def register_pull_handler(self, name: str, func: Callable):
        """Register UDF pull function."""
        raise NotImplementedError("register_pull_handler is not implemented for the ops-based client.")

    def register_tensor_meta(
        self,
        name: str,
        shape: Tuple[int, int],
        dtype: torch.dtype,
        base_offset: int = 0,
    ) -> None:
        if name in self._tensor_meta:
            return
        normalized_shape = (int(shape[0]), int(shape[1]))
        self._tensor_meta[name] = {
            "shape": normalized_shape,
            "dtype": dtype,
            "base_offset": int(base_offset),
            "tag": 0,
            "table_id": 0,
        }
        self._full_data_shape[name] = normalized_shape
        self._data_name_list.add(name)
        self._gdata_name_list.add(name)

    def init_embedding_table(
        self,
        table_name: str,
        num_embeddings: int,
        embedding_dim: int,
        table_id: Optional[int] = None,
    ) -> int:
        if table_id is None:
            table_id = self._next_table_id
        table_id = int(table_id)
        if table_id < 0:
            raise ValueError("table_id must be non-negative")
        self._next_table_id = max(self._next_table_id, table_id + 1)
        tag = int(
            self.ops.init_embedding_table(
                table_name,
                int(num_embeddings),
                int(embedding_dim),
                table_id,
            )
        )
        if tag < 0:
            return -1
        if table_name in self._tensor_meta:
            self._tensor_meta[table_name]["tag"] = tag
            self._tensor_meta[table_name]["table_id"] = table_id
        return tag

    def init_data(self, name: str, shape: Tuple[int, int], dtype: torch.dtype, part_policy: Any = None, init_func: Optional[Callable] = None, is_gdata: bool = True, base_offset: int = 0):
        """Send message to kvserver to initialize new data tensor and mapping this
        data from server side to client side.

        Parameters
        ----------
        name : str
            data name
        shape : list or tuple of int
            data shape
        dtype : dtype
            data type
        part_policy : PartitionPolicy
            partition policy.
        init_func : func
            UDF init function
        is_gdata : bool
            Whether the created tensor is a ndata/edata or not.
        """
        if name in self._tensor_meta:
            print(f"Tensor '{name}' already exists. Skipping initialization.")
            return

        # print(f"Initializing tensor '{name}' with shape {shape} and dtype {dtype} (base_offset={base_offset}).")
        
        num_embeddings, embedding_dim = shape
        table_id = self._next_table_id
        tag = self.init_embedding_table(
            name, num_embeddings, embedding_dim, table_id
        )
        self._clear_gpu_cache_if_available()
        if tag < 0:
            raise RuntimeError(f"Failed to initialize embedding table '{name}' on backend.")
        
        self.register_tensor_meta(
            name=name,
            shape=shape,
            dtype=dtype,
            base_offset=base_offset,
        )
        self._tensor_meta[name]["tag"] = tag
        self._tensor_meta[name]["table_id"] = table_id
        if not is_gdata:
            self._gdata_name_list.discard(name)
        
        # Avoid materializing a full dense tensor for large embedding tables
        # unless the caller explicitly requests custom initialization data.
        if init_func is not None:
            initial_data = init_func(shape, dtype)
        else:
            initial_data = torch.zeros(shape, dtype=dtype)
        
        all_keys = torch.arange(shape[0], dtype=torch.int64)
        if base_offset != 0:
            all_keys = all_keys + int(base_offset)
        self.ops.emb_write(self._encode_ids(name, all_keys), initial_data)
        self._clear_gpu_cache_if_available()


    def delete_data(self, name: str):
        """Send message to kvserver to delete tensor and clear the meta data

        Parameters
        ----------
        name : str
            data name
        """
        if name not in self._tensor_meta:
            print(f"Warning: Tensor '{name}' does not exist. Cannot delete.")
            return
        
        del self._tensor_meta[name]
        del self._full_data_shape[name]
        self._data_name_list.remove(name)
        if name in self._gdata_name_list:
            self._gdata_name_list.remove(name)
        
        raise NotImplementedError("delete_data is not fully implemented for the ops-based client; backend data is not cleared.")

    def map_shared_data(self, partition_book: Any):
        """Mapping shared-memory tensor from server to client.

        Parameters
        ----------
        partition_book : GraphPartitionBook
            Store the partition information
        """
        raise NotImplementedError("map_shared_data is not applicable for the ops-based client.")

    def gdata_name_list(self) -> List[str]:
        """Get all the graph data name"""
        return list(self._gdata_name_list)

    def get_partid(self, name: str, id_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor
            a vector storing the global data ID
        """
        raise NotImplementedError("get_partid is not applicable in a non-partitioned, ops-based implementation.")

    def pull(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Pull message from KVServer.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor
            a vector storing the ID list

        Returns
        -------
        tensor
            a data tensor with the same row size of id_tensor.
        """
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        
        meta = self._tensor_meta[name]
        embedding_dim = meta['shape'][1]
        ids = self._normalize_ids(ids, name=name)
            
        start_t = time.time()
        res = self.ops.emb_read(ids, embedding_dim)
        end_t = time.time()
        
        start_us = int(start_t * 1e6)
        duration_us = (end_t - start_t) * 1e6
        report_metric("embread_stages", f"KVClient::pull|{start_us}", "duration_us", duration_us)
        report_metric("embread_stages", f"KVClient::pull|{start_us}", "request_size", float(ids.numel()))
        
        return res

    def pull_with_gpu_cache(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Pull rows while preserving CUDA ids so the backend can use GPU cache."""
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")

        meta = self._tensor_meta[name]
        embedding_dim = meta['shape'][1]
        ids = self._normalize_ids(ids, preserve_device=True, name=name)
        self._reject_gpu_cache_reserved_ids(ids)

        start_t = time.time()
        res = self.ops.emb_read(ids, embedding_dim)
        end_t = time.time()

        start_us = int(start_t * 1e6)
        duration_us = (end_t - start_t) * 1e6
        report_metric("embread_stages", f"KVClient::pull_with_gpu_cache|{start_us}", "duration_us", duration_us)
        report_metric(
            "embread_stages",
            f"KVClient::pull_with_gpu_cache|{start_us}",
            "request_size",
            float(ids.numel()),
        )

        return res

    def _normalize_ids(
        self,
        ids: torch.Tensor,
        *,
        preserve_device: bool = False,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")
        if ids.dtype != torch.int64:
            ids = ids.to(dtype=torch.int64)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        if preserve_device and ids.device.type not in ("cpu", "cuda"):
            raise RuntimeError(
                f"local_shm fast path only supports cpu or cuda tensors, got {ids.device.type}."
            )
        if not preserve_device and ids.device.type != 'cpu':
            ids = ids.to('cpu')
        if name is not None:
            ids = self._encode_ids(name, ids)
        return ids

    def _encode_ids(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        meta = self._tensor_meta.get(name)
        tag = int(meta.get("tag", 0) or 0) if meta else 0
        if tag == 0 or ids.numel() == 0:
            return ids
        existing = ids.bitwise_right_shift(_KEY_TAG_SHIFT)
        if bool((existing == tag).all().item()):
            return ids
        if not bool((existing == 0).all().item()):
            raise RuntimeError(
                f"ids for '{name}' already carry tag {existing.unique().tolist()}, expected {tag}"
            )
        tag_bits = torch.tensor(
            tag << _KEY_TAG_SHIFT, dtype=ids.dtype, device=ids.device
        )
        return ids.bitwise_or(tag_bits)

    def _reject_gpu_cache_reserved_ids(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        if ids.device.type == "cuda" and os.getenv(
            "RECSTORE_VALIDATE_GPU_CACHE_KEYS", ""
        ) not in ("1", "true", "TRUE", "yes", "YES"):
            return
        empty_key = torch.iinfo(torch.int64).max
        deleted_key = empty_key - 1
        if int(ids.max().item()) >= deleted_key:
            raise RuntimeError(
                "ids contain reserved GPU cache sentinel key; "
                f"values {deleted_key} and {empty_key} are not valid RecStore GPU cache keys"
            )

    def _normalize_grads(
        self,
        grads: torch.Tensor,
        *,
        preserve_device: bool = False,
    ) -> torch.Tensor:
        if not isinstance(grads, torch.Tensor):
            raise TypeError("grads must be a torch.Tensor")
        if grads.dtype != torch.float32:
            grads = grads.to(dtype=torch.float32)
        if not grads.is_contiguous():
            grads = grads.contiguous()
        if preserve_device and grads.device.type not in ("cpu", "cuda"):
            raise RuntimeError(
                f"local_shm fast path only supports cpu or cuda tensors, got {grads.device.type}."
            )
        if not preserve_device and grads.device.type != 'cpu':
            grads = grads.to('cpu')
        return grads

    def _require_local_shm_backend(self, api_name: str) -> None:
        backend = self.current_ps_backend()
        if backend not in _LOCAL_FAST_PATH_BACKENDS:
            raise RuntimeError(
                f"{api_name} requires local_shm or hierkv backend, but current backend is {backend}."
            )

    def set_ps_backend(self, backend: str) -> None:
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string")
        self.ops.set_ps_backend(backend)

    def current_ps_backend(self) -> str:
        return str(self.ops.current_ps_backend())

    def is_shared_local_shm_table(self) -> bool:
        return self.current_ps_backend().lower() == "local_shm"

    def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._require_local_shm_backend("local_lookup_flat")
        self._ensure_gpu_cache_table(name)
        meta = self._tensor_meta[name]
        embedding_dim = meta['shape'][1]
        ids = self._normalize_ids(ids, preserve_device=True, name=name)
        self._reject_gpu_cache_reserved_ids(ids)
        return self.ops.local_lookup_flat(ids, int(embedding_dim))

    def warmup_local_lookup_flat_cuda_region(self) -> bool:
        self._require_local_shm_backend("warmup_local_lookup_flat_cuda_region")
        warmup = getattr(self.ops, "warmup_local_lookup_flat_cuda_region", None)
        if not callable(warmup):
            return False
        return bool(warmup())

    def enable_gpu_cache(self, capacity: int, embedding_dim: int) -> bool:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError("capacity must be an integer")
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool):
            raise ValueError("embedding_dim must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._gpu_cache_enabled = bool(
            self.ops.enable_gpu_cache(int(capacity), int(embedding_dim))
        )
        return self._gpu_cache_enabled

    def is_gpu_cache_enabled(self) -> bool:
        return bool(self._gpu_cache_enabled)

    def disable_gpu_cache(self) -> None:
        self.ops.disable_gpu_cache()
        self._gpu_cache_enabled = False

    def clear_gpu_cache(self) -> None:
        self.ops.clear_gpu_cache()
        self._gpu_cache_clear_count += 1
        self._gpu_cache_table_name = None

    def prefill_gpu_cache(self, name: str, ids: torch.Tensor, values: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        ids = self._normalize_ids(ids, preserve_device=True, name=name)
        if values.dim() != 2:
            raise ValueError("values must be a 2-dimensional tensor")
        if ids.size(0) != values.size(0):
            raise ValueError("ids and values must have the same number of rows")
        if ids.device.type == "cpu":
            self._reject_gpu_cache_reserved_ids(ids)
        self._ensure_gpu_cache_table(name)
        self.ops.prefill_gpu_cache(ids, values)

    def invalidate_gpu_cache(self, name: str, ids: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._ensure_gpu_cache_table(name)
        ids = self._normalize_ids(ids, preserve_device=True, name=name)
        if ids.device.type == "cpu":
            if torch.cuda.is_available():
                ids = ids.to(torch.device("cuda", torch.cuda.current_device()))
            else:
                raise RuntimeError("invalidate_gpu_cache requires CUDA ids")
        self.ops.invalidate_gpu_cache(ids)

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
        self._ensure_gpu_cache_table(name)
        ids = self._normalize_ids(ids, preserve_device=True, name=name)
        grads = self._normalize_grads(grads, preserve_device=True)
        if grads.dim() != 2:
            raise ValueError("grads must be a 2-dimensional tensor")
        if ids.size(0) != grads.size(0):
            raise ValueError("ids and grads must have the same number of rows")
        if ids.device.type == "cpu":
            self._reject_gpu_cache_reserved_ids(ids)
        return bool(self.ops.apply_sgd_update_gpu_cache(ids, grads, float(learning_rate)))

    def set_gpu_cache_lookup_bypass_enabled(self, enabled: bool) -> None:
        self.ops.set_gpu_cache_lookup_bypass_enabled(bool(enabled))

    def is_gpu_cache_lookup_bypass_enabled(self) -> bool:
        return bool(self.ops.is_gpu_cache_lookup_bypass_enabled())

    def is_gpu_cache_lookup_bypassed(self) -> bool:
        return bool(self.ops.is_gpu_cache_lookup_bypassed())

    def reset_gpu_cache_bypass_state(self) -> None:
        self.ops.reset_gpu_cache_bypass_state()

    def _clear_gpu_cache_if_available(self) -> None:
        clear = getattr(self.ops, "clear_gpu_cache", None)
        if callable(clear):
            clear()
            self._gpu_cache_clear_count += 1
        self._gpu_cache_table_name = None

    def get_gpu_cache_clear_count(self) -> int:
        return int(self._gpu_cache_clear_count)

    def set_clear_gpu_cache_after_cpu_update(self, enabled: bool) -> None:
        self._clear_gpu_cache_after_cpu_update = bool(enabled)

    def set_prefetch_table_name(self, name: str) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._prefetch_table_name = name

    def _ensure_gpu_cache_table(self, name: str) -> None:
        if self._gpu_cache_table_name is None:
            self._gpu_cache_table_name = name
            return
        if self._gpu_cache_table_name != name:
            self._clear_gpu_cache_if_available()
            self._gpu_cache_table_name = name

    def gpu_cache_lookup_flat(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """GPU-cache-accelerated flat lookup that works with any backend.

        Serves cache hits from the GPU cache; misses are fetched via the
        active backend (BRPC/GRPC/RDMA/local_shm) and filled back into the
        cache.  Returns a [num_keys, embedding_dim] float32 tensor on the
        same device as ``keys`` (CUDA when keys are CUDA).
        """
        keys = self._normalize_ids(
            keys, preserve_device=True, name=self._gpu_cache_table_name
        )
        if not keys.is_contiguous():
            keys = keys.contiguous()
        return self.ops.gpu_cache_lookup_flat(keys, int(embedding_dim))

    def query_gpu_cache(self, keys: torch.Tensor, embedding_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Query GPU cache. Returns (values, missing_keys).

        values has shape [num_keys, embedding_dim] with zeros for misses.
        missing_keys is a CPU int64 tensor of keys not found in cache."""
        result = self.ops.query_gpu_cache(keys, int(embedding_dim))
        return result[0], result[1]

    def update_gpu_cache(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        """Update existing GPU cache entries with new values (no eviction)."""
        self._ensure_gpu_cache_table(self._gpu_cache_table_name or "")
        ids = self._normalize_ids(ids, preserve_device=True, name=self._gpu_cache_table_name)
        values = self._normalize_grads(values, preserve_device=True)
        self.ops.update_gpu_cache(ids, values)

    def emb_write_values(self, name: str, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Write (set) embedding values directly to the PS for a subset of
        keys, with per-key GPU cache invalidation (no full cache clear).

        Used by the BagPipe eviction writeback path to push locally-updated
        cache values back to the PS without disturbing other cached entries.
        """
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        if keys.numel() == 0:
            return
        self._ensure_gpu_cache_table(name)
        ids = self._normalize_ids(keys, preserve_device=True, name=name)
        if ids.device.type == "cpu":
            if torch.cuda.is_available():
                ids = ids.to(torch.device("cuda", torch.cuda.current_device()))
            else:
                raise RuntimeError("emb_write_values requires CUDA ids")
        vals = self._normalize_grads(values, preserve_device=True)
        if vals.device.type == "cpu" and torch.cuda.is_available():
            vals = vals.to(ids.device)
        self.ops.emb_write_values(ids, vals)

    def local_update_flat(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")

        self._require_local_shm_backend("local_update_flat")
        self._ensure_gpu_cache_table(name)
        ids = self._normalize_ids(ids, preserve_device=True, name=name)
        if grads.dim() != 2:
            raise ValueError("grads must be a 2-dimensional tensor")
        if ids.size(0) != grads.size(0):
            raise ValueError("ids and grads must have the same number of rows")
        if ids.device.type == "cpu":
            self._reject_gpu_cache_reserved_ids(ids)
        self.ops.local_update_flat(name, ids, grads)
        if ids.device.type == "cpu":
            self._clear_gpu_cache_if_available()

    def push(self, name: str, ids: torch.Tensor, data: torch.Tensor):
        """Push data to KVServer.

        Note that, the push() is an non-blocking operation that will return immediately.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor
            a vector storing the global data ID
        data_tensor : tensor
            a tensor with the same row size of data ID
        """
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self.ops.emb_write(self._normalize_ids(ids, name=name), data)
        self._clear_gpu_cache_if_available()

    # ---- Prefetch APIs ----
    def prefetch(self, ids: torch.Tensor, name: Optional[str] = None) -> int:
        """Initiate an async prefetch for given ids. Returns a handle (int).

        The returned handle should be consumed soon (same batch) to avoid cache pressure.
        """
        table_name = name or self._prefetch_table_name
        ids = self._normalize_ids(ids, name=table_name)
        return int(self.ops.emb_prefetch(ids))

    def wait_and_get(self, prefetch_id: int, embedding_dim: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Block until prefetch completes and return embeddings of shape [N, D]."""
        start_t = time.time()
        out = self.ops.emb_wait_result(int(prefetch_id), int(embedding_dim))
        end_t = time.time()
        
        start_us = int(start_t * 1e6)
        duration_us = (end_t - start_t) * 1e6
        report_metric("embread_stages", f"KVClient::wait_and_get|{start_us}", "duration_us", duration_us)
        
        if device.type == "cuda":
            out = out.to(device)
        return out

    def update(self, name: str, ids: torch.Tensor, grads: torch.Tensor):
        """
        Pushes gradients to update the given IDs of a named tensor via embupdate.
        Backend optimizers apply their own learning rate when consuming these grads.
        Callers should pass aggregated raw gradients unless they are intentionally
        targeting a backend path that expects pre-scaled values.
        """
        handle = self.update_async(name, ids, grads)
        self.wait(handle)

    def update_async(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> int:
        """Submit an embedding update and return a handle for synchronization."""
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        
        ids = self._normalize_ids(ids, name=name)
        grads = self._normalize_grads(grads)

        handle = self._next_async_handle
        self._next_async_handle += 1
        # ponytail: soft-optional RDMA async; staged fallback if ops lack it
        submit = getattr(self.ops, "emb_update_async", None)
        if (
            hasattr(self.ops, "current_ps_backend")
            and self.ops.current_ps_backend() == "rdma"
            and callable(submit)
        ):
            backend_handle = int(submit(name, ids, grads))
            self._pending_async_ops[handle] = ("rdma", backend_handle)
        else:
            self._pending_async_ops[handle] = (
                "staged",
                name,
                ids.clone(),
                grads.clone(),
            )
        return handle

    def wait(self, handle: int) -> None:
        """Wait for a queued async operation and apply it if still pending."""
        pending = self._pending_async_ops.pop(int(handle), None)
        if pending is None:
            return
        if pending[0] == "rdma":
            self.ops.emb_update_wait(int(pending[1]))
            return
        _, name, ids, grads = pending
        self.ops.emb_update_table(name, ids, grads)

    def flush_async_updates(self) -> None:
        """Synchronously apply all queued async update operations."""
        for handle in list(self._pending_async_ops.keys()):
            self.wait(handle)

    def get_data_meta(self, name: str) -> Tuple[torch.dtype, Tuple[int, ...], None]:
        """Get meta data (data_type, data_shape, partition_policy)"""
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' does not exist.")
        meta = self._tensor_meta[name]
        return meta['dtype'], meta['shape']
        # part_policy = self._part_policy[name]
        # return meta['dtype'], self._full_data_shape[name], part_policy

    def data_name_list(self) -> List[str]:
        """Get all the data name"""
        return list(self._tensor_meta.keys())

    def count_nonzero(self, name: str) -> int:
        """Count nonzero value by pull request from KVServers.

        Parameters
        ----------
        name : str
            data name

        Returns
        -------
        int
            the number of nonzero in this data.
        """
        raise NotImplementedError("count_nonzero is not implemented for the ops-based client.")

    def set_ps_config(self, host: str, port: int):
        """
        Dynamically configure the PS Client host and port.
        This forces re-initialization of the backend PS client.
        """
        print(f"[RecStoreClient] Setting PS config to {host}:{port}")
        self.ops.set_ps_config(host, int(port))

def get_kv_client() -> RecStoreClient:
    """
    Factory function to get the singleton instance of the RecStoreClient.
    """
    return RecStoreClient()
