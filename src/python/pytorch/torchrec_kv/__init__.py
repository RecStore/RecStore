"""torchrec_kv package: embedding bag and collection implementations."""

from .EmbeddingBag import RecStoreEmbeddingBagCollection
from .EmbeddingCollection import RecStoreEmbeddingCollection

__all__ = [
    "RecStoreEmbeddingBagCollection",
    "RecStoreEmbeddingCollection",
]
