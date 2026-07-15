import torch
import os
import time
import ctypes
from typing import Optional, Tuple, List, Dict, Any, Callable

_LOCAL_FAST_PATH_BACKENDS = {"local_shm", "hierkv"}
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

    def __init__(self, library_path: Optional[str] = None, role: str = "default"):
        if self._initialized:
            return
        
        if library_path is None:
            script_dir = os.path.dirname(__file__)
            default_lib_path = os.path.abspath(os.path.join(script_dir, '../../../../build/lib/lib_recstore_ops.so'))
            if not os.path.exists(default_lib_path):
                 raise ImportError(
                    f"Could not find Recstore library at default path: {default_lib_path}\n"
                    "Please provide the correct path or ensure your project is built correctly."
                )
            library_path = default_lib_path
        
        torch.ops.load_library(library_path)
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
        self._initialized = True
        # print(f"RecStoreClient initialized. Loaded library from: {library_path}")

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
        # print(f"[DEBUG] Calling init_embedding_table for '{name}' with num_embeddings={num_embeddings}, embedding_dim={embedding_dim}")
        success = self.ops.init_embedding_table(name, int(num_embeddings), int(embedding_dim))
        self._clear_gpu_cache_if_available()
        # print(f"[DEBUG] init_embedding_table returned: {success}")
        if not success:
            raise RuntimeError(f"Failed to initialize embedding table '{name}' on backend.")
        
        self._tensor_meta[name] = {'shape': shape, 'dtype': dtype}
        self._full_data_shape[name] = shape
        self._data_name_list.add(name)
        if is_gdata:
            self._gdata_name_list.add(name)
        
        # Avoid materializing a full dense tensor for large embedding tables
        # unless the caller explicitly requests custom initialization data.
        if init_func is not None:
            initial_data = init_func(shape, dtype)
        else:
            initial_data = torch.zeros(shape, dtype=dtype)
        
        all_keys = torch.arange(shape[0], dtype=torch.int64)
        if base_offset != 0:
            all_keys = all_keys + int(base_offset)
        self.ops.emb_write(all_keys, initial_data)
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
        
        # Ensure ids are on the correct device and dtype
        # Normalize ids: force CPU, int64, contiguous
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")
        if ids.dtype != torch.int64:
            ids = ids.to(dtype=torch.int64)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        # Some upstream code may pass CUDA tensors; backend ops are CPU-only.
        if ids.device.type != 'cpu':
            ids = ids.to('cpu')
            
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
        ids = self._normalize_ids(ids, preserve_device=True)
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
        return ids

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
        if not hasattr(self.ops, "current_ps_backend"):
            raise RuntimeError(
                f"{api_name} requires a RecStore ops library exposing current_ps_backend()."
            )
        backend = self.ops.current_ps_backend()
        if backend not in _LOCAL_FAST_PATH_BACKENDS:
            raise RuntimeError(
                f"{api_name} requires local_shm or hierkv backend, but current backend is {backend}."
            )

    def set_ps_backend(self, backend: str) -> None:
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string")
        if not hasattr(self.ops, "set_ps_backend"):
            raise RuntimeError(
                "set_ps_backend requires a RecStore ops library exposing set_ps_backend()."
            )
        self.ops.set_ps_backend(backend)

    def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._require_local_shm_backend("local_lookup_flat")
        self._ensure_gpu_cache_table(name)
        meta = self._tensor_meta[name]
        embedding_dim = meta['shape'][1]
        ids = self._normalize_ids(ids, preserve_device=True)
        self._reject_gpu_cache_reserved_ids(ids)
        return self.ops.local_lookup_flat(ids, int(embedding_dim))

    def get_last_local_shm_lookup_profile(self) -> Dict[str, float]:
        getter = getattr(self.ops, "get_last_local_lookup_flat_profile", None)
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

    def enable_gpu_cache(self, capacity: int, embedding_dim: int) -> bool:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError("capacity must be an integer")
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool):
            raise ValueError("embedding_dim must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        enable = getattr(self.ops, "enable_gpu_cache", None)
        if not callable(enable):
            raise RuntimeError(
                "enable_gpu_cache requires a RecStore ops library exposing enable_gpu_cache()."
            )
        return bool(enable(int(capacity), int(embedding_dim)))

    def disable_gpu_cache(self) -> None:
        disable = getattr(self.ops, "disable_gpu_cache", None)
        if not callable(disable):
            raise RuntimeError(
                "disable_gpu_cache requires a RecStore ops library exposing disable_gpu_cache()."
            )
        disable()

    def clear_gpu_cache(self) -> None:
        clear = getattr(self.ops, "clear_gpu_cache", None)
        if not callable(clear):
            raise RuntimeError(
                "clear_gpu_cache requires a RecStore ops library exposing clear_gpu_cache()."
            )
        clear()
        self._gpu_cache_table_name = None

    def prefill_gpu_cache(self, name: str, ids: torch.Tensor, values: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        prefill = getattr(self.ops, "prefill_gpu_cache", None)
        if not callable(prefill):
            raise RuntimeError(
                "prefill_gpu_cache requires a RecStore ops library exposing "
                "prefill_gpu_cache()."
            )
        self._ensure_gpu_cache_table(name)
        ids = self._normalize_ids(ids, preserve_device=True)
        values = self._normalize_grads(values, preserve_device=True)
        if values.dim() != 2:
            raise ValueError("values must be a 2-dimensional tensor")
        if ids.size(0) != values.size(0):
            raise ValueError("ids and values must have the same number of rows")
        if ids.device.type == "cpu":
            self._reject_gpu_cache_reserved_ids(ids)
        prefill(ids, values)

    def invalidate_gpu_cache(self, name: str, ids: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        invalidator = getattr(self.ops, "invalidate_gpu_cache", None)
        if not callable(invalidator):
            raise RuntimeError(
                "invalidate_gpu_cache requires a RecStore ops library exposing "
                "invalidate_gpu_cache()."
            )
        self._ensure_gpu_cache_table(name)
        ids = self._normalize_ids(ids, preserve_device=True)
        if ids.device.type == "cpu":
            if torch.cuda.is_available():
                ids = ids.to(torch.device("cuda", torch.cuda.current_device()))
            else:
                raise RuntimeError("invalidate_gpu_cache requires CUDA ids")
        invalidator(ids)

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
        updater = getattr(self.ops, "apply_sgd_update_gpu_cache", None)
        if not callable(updater):
            raise RuntimeError(
                "apply_sgd_update_gpu_cache requires a RecStore ops library exposing "
                "apply_sgd_update_gpu_cache()."
            )
        self._ensure_gpu_cache_table(name)
        ids = self._normalize_ids(ids, preserve_device=True)
        grads = self._normalize_grads(grads, preserve_device=True)
        if grads.dim() != 2:
            raise ValueError("grads must be a 2-dimensional tensor")
        if ids.size(0) != grads.size(0):
            raise ValueError("ids and grads must have the same number of rows")
        if ids.device.type == "cpu":
            self._reject_gpu_cache_reserved_ids(ids)
        return bool(updater(ids, grads, float(learning_rate)))

    def set_gpu_cache_lookup_bypass_enabled(self, enabled: bool) -> None:
        setter = getattr(self.ops, "set_gpu_cache_lookup_bypass_enabled", None)
        if not callable(setter):
            raise RuntimeError(
                "set_gpu_cache_lookup_bypass_enabled requires a RecStore ops library "
                "exposing set_gpu_cache_lookup_bypass_enabled()."
            )
        setter(bool(enabled))

    def is_gpu_cache_lookup_bypass_enabled(self) -> bool:
        getter = getattr(self.ops, "is_gpu_cache_lookup_bypass_enabled", None)
        if not callable(getter):
            raise RuntimeError(
                "is_gpu_cache_lookup_bypass_enabled requires a RecStore ops library "
                "exposing is_gpu_cache_lookup_bypass_enabled()."
            )
        return bool(getter())

    def is_gpu_cache_lookup_bypassed(self) -> bool:
        getter = getattr(self.ops, "is_gpu_cache_lookup_bypassed", None)
        if not callable(getter):
            raise RuntimeError(
                "is_gpu_cache_lookup_bypassed requires a RecStore ops library "
                "exposing is_gpu_cache_lookup_bypassed()."
            )
        return bool(getter())

    def reset_gpu_cache_bypass_state(self) -> None:
        reset = getattr(self.ops, "reset_gpu_cache_bypass_state", None)
        if not callable(reset):
            raise RuntimeError(
                "reset_gpu_cache_bypass_state requires a RecStore ops library "
                "exposing reset_gpu_cache_bypass_state()."
            )
        reset()

    def _clear_gpu_cache_if_available(self) -> None:
        clear = getattr(self.ops, "clear_gpu_cache", None)
        if callable(clear):
            clear()
        self._gpu_cache_table_name = None

    def _ensure_gpu_cache_table(self, name: str) -> None:
        if self._gpu_cache_table_name is None:
            self._gpu_cache_table_name = name
            return
        if self._gpu_cache_table_name != name:
            self._clear_gpu_cache_if_available()
            self._gpu_cache_table_name = name

    def get_last_gpu_cache_profile(self) -> Dict[str, float]:
        getter = getattr(self.ops, "get_last_gpu_cache_profile", None)
        if not callable(getter):
            return {}
        values = getter()
        if not isinstance(values, (list, tuple)) or len(values) < 5:
            return {}
        profile = {}
        for index, key in enumerate(_GPU_CACHE_PROFILE_KEYS):
            profile[key] = float(values[index]) if index < len(values) else 0.0
        return profile

    def local_update_flat(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")

        self._require_local_shm_backend("local_update_flat")
        ids = self._normalize_ids(ids, preserve_device=True)
        grads = self._normalize_grads(grads, preserve_device=True)
        self._ensure_gpu_cache_table(name)
        if grads.dim() != 2:
            raise ValueError("grads must be a 2-dimensional tensor")
        if ids.size(0) != grads.size(0):
            raise ValueError("ids and grads must have the same number of rows")
        if ids.device.type == "cpu":
            self._reject_gpu_cache_reserved_ids(ids)
        self.ops.local_update_flat(name, ids, grads)
        if ids.device.type == "cpu":
            self._clear_gpu_cache_if_available()

    def get_last_local_shm_update_profile(self) -> Dict[str, float]:
        getter = getattr(self.ops, "get_last_local_update_flat_profile", None)
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
        self.ops.emb_write(ids, data)
        self._clear_gpu_cache_if_available()

    # ---- Prefetch APIs ----
    def prefetch(self, ids: torch.Tensor) -> int:
        """Initiate an async prefetch for given ids. Returns a handle (int).

        The returned handle should be consumed soon (same batch) to avoid cache pressure.
        """
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")
        if ids.dtype != torch.int64:
            ids = ids.to(dtype=torch.int64)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        if ids.device.type != 'cpu':
            ids = ids.to('cpu')
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
        """Queue an embedding update and return a handle for explicit synchronization."""
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        
        ids = self._normalize_ids(ids)
        grads = self._normalize_grads(grads)

        handle = self._next_async_handle
        self._next_async_handle += 1
        self._pending_async_ops[handle] = (
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
        name, ids, grads = pending
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
