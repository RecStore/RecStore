"""Structural protocols for optimization plugins.

These ``typing.Protocol`` subclasses define the interface that optimization
plugins depend on.  They are *structural* (duck-typed) — existing classes
(:class:`~recstore.sharded_client.ShardedRecstoreClient`,
``RecStoreEmbeddingBagCollection``) satisfy them without inheritance, and
runtime overhead is zero.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KVClientProtocol(Protocol):
    """KV client interface required by optimization plugins.

    ``ShardedRecstoreClient`` and ``RecStoreClient`` both satisfy this
    protocol.
    """

    def prefetch(self, ids: Any) -> int: ...
    def wait_and_get(self, prefetch_id: int, embedding_dim: int, device: Any = ...) -> Any: ...
    def update(self, name: str, ids: Any, grads: Any) -> None: ...
    def update_async(self, name: str, ids: Any, grads: Any) -> int: ...
    def wait(self, handle: int) -> None: ...
    def emb_write_values(self, name: str, keys: Any, values: Any) -> None: ...
    def prefill_gpu_cache(self, name: str, ids: Any, values: Any) -> None: ...
    def invalidate_gpu_cache(self, name: str, ids: Any) -> None: ...
    def apply_sgd_update_gpu_cache(self, *args: Any, **kwargs: Any) -> None: ...
    def gpu_cache_lookup_flat(self, keys: Any, embedding_dim: int) -> Any: ...


@runtime_checkable
class EmbeddingModuleProtocol(Protocol):
    """Embedding module interface required by optimization plugins.

    ``RecStoreEmbeddingBagCollection`` satisfies this protocol.
    """

    def issue_fused_prefetch(self, sparse_features: Any, **kwargs: Any) -> Any: ...
    def set_fused_prefetch_handle(self, handle: int, **kwargs: Any) -> None: ...
    def __call__(self, sparse_features: Any) -> Any: ...
