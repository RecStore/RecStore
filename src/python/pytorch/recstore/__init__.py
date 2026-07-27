import os
from pathlib import Path

import torch


def load_ops_library() -> None:
    torch.ops.load_library(
        os.environ.get(
            "RECSTORE_OPS_LIBRARY",
            str(Path(__file__).resolve().parents[4] / "build/lib/lib_recstore_ops.so"),
        )
    )


if os.environ.get("RECSTORE_DEFER_OPS_LOAD") != "1":
    load_ops_library()

# ---------------------------------------------------------------------------
# Pure-Python submodules
# ---------------------------------------------------------------------------
from .DistTensor import DistTensor  # noqa: E402,F401
from .DistEmb import DistEmbedding  # noqa: E402,F401
from .KVClient import RecStoreClient, get_kv_client  # noqa: E402,F401
from .optimizer import SparseSGD  # noqa: E402,F401
# Import optim before bagpipe_cache: optim's _register_builtins() imports
# bagpipe_cache.plugin, which in turn imports recstore.optim.plugin.
# This ordering ensures optim.plugin is fully loaded when bagpipe_cache loads.
from . import optim  # noqa: E402,F401
from . import bagpipe_cache  # noqa: E402,F401

# Lazy: importing EmbeddingBag pulls recstore.KVClient and would re-enter this
# package while torchrec_kv.EmbeddingBag is still initializing (circular import
# when tests load via src.python.pytorch.recstore.* with PYTHONPATH set).
def __getattr__(name: str):
    if name == "RecStoreEmbeddingBagCollection":
        from torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection as _cls

        globals()[name] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DistTensor",
    "DistEmbedding",
    "RecStoreClient",
    "get_kv_client",
    "SparseSGD",
    "bagpipe_cache",
    "optim",
    "RecStoreEmbeddingBagCollection",
    "load_ops_library",
]
