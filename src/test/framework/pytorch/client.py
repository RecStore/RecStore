import os
from typing import Optional

import torch


class RecstoreClient:
    _is_initialized = False

    def __init__(self, library_path: Optional[str] = None):
        if RecstoreClient._is_initialized:
            return

        if library_path is None:
            script_dir = os.path.dirname(__file__)
            default_lib_path = os.path.abspath(
                os.path.join(script_dir, "../../../../build/_recstore_ops.so")
            )
            if not os.path.exists(default_lib_path):
                raise ImportError(
                    f"Could not find Recstore library at default path: {default_lib_path}\n"
                    "Please provide the correct path via the 'library_path' argument "
                    "or ensure your project is built correctly with build.sh."
                )
            library_path = default_lib_path

        torch.ops.load_library(library_path)
        self.ops = torch.ops.recstore_ops
        RecstoreClient._is_initialized = True

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        return self.ops.emb_read(keys, embedding_dim)

    def emb_update(self, keys: torch.Tensor, grads: torch.Tensor) -> None:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if grads.dtype != torch.float32:
            raise TypeError(f"grads tensor must be of dtype torch.float32, but got {grads.dtype}")
        if keys.shape[0] != grads.shape[0]:
            raise ValueError("keys and grads must have the same number of entries")
        self.ops.emb_update_table("default", keys, grads)

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        if not table_name:
            raise ValueError("table_name must be non-empty")
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if grads.dtype != torch.float32:
            raise TypeError(f"grads tensor must be of dtype torch.float32, but got {grads.dtype}")
        if keys.shape[0] != grads.shape[0]:
            raise ValueError("keys and grads must have the same number of entries")

        self.ops.emb_update_table(table_name, keys, grads)

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        if not table_name:
            raise ValueError("table_name must be non-empty")
        if not isinstance(num_embeddings, int) or num_embeddings <= 0:
            raise ValueError("num_embeddings must be a positive integer")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        return bool(self.ops.init_embedding_table(table_name, num_embeddings, embedding_dim))

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if values.dtype != torch.float32:
            raise TypeError(f"values tensor must be of dtype torch.float32, but got {values.dtype}")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have the same number of entries")

        self.ops.emb_write(keys, values)

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        pid = self.ops.emb_prefetch(keys)
        return int(pid)

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        if not isinstance(prefetch_id, int) or prefetch_id <= 0:
            raise ValueError("prefetch_id must be a positive int")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive int")
        return self.ops.emb_wait_result(int(prefetch_id), int(embedding_dim))

    # ---- BagPipe-style GPU cache ops (backend-agnostic) ----
    def gpu_cache_lookup_flat(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        return self.ops.gpu_cache_lookup_flat(keys, int(embedding_dim))

    def query_gpu_cache(self, keys: torch.Tensor, embedding_dim: int):
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        return self.ops.query_gpu_cache(keys, int(embedding_dim))

    def update_gpu_cache(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        if ids.dtype != torch.int64:
            raise TypeError(f"ids tensor must be of dtype torch.int64, but got {ids.dtype}")
        if values.dtype != torch.float32:
            raise TypeError(f"values tensor must be of dtype torch.float32, but got {values.dtype}")
        if ids.shape[0] != values.shape[0]:
            raise ValueError("ids and values must have the same number of entries")
        self.ops.update_gpu_cache(ids, values)

    def invalidate_gpu_cache(self, keys: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        self.ops.invalidate_gpu_cache(keys)

    def apply_sgd_update_gpu_cache(self, keys: torch.Tensor, grads: torch.Tensor, lr: float) -> bool:
        if keys.numel() == 0:
            return True
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if grads.dtype != torch.float32:
            raise TypeError(f"grads tensor must be of dtype torch.float32, but got {grads.dtype}")
        return bool(self.ops.apply_sgd_update_gpu_cache(keys, grads, float(lr)))
