"""RecStore: distributed parameter server and embedding store.

Importing this package automatically loads the native extension
``recstore._recstore_ops`` (a single shared object that registers the
``torch.ops.recstore_ops`` namespace via ``TORCH_LIBRARY``).  The loader
searches, in order:

    1. Wheel install location:  recstore/lib/_recstore_ops.so
    2. In-place dev build:      recstore/_recstore_ops.so
    3. Legacy CMake artifacts:  build/lib/lib_recstore_ops.so
    4. setuptools build dir:   build/lib.*/recstore/_recstore_ops.so
                                 or build/lib.*/recstore/lib/_recstore_ops.so

Loading is idempotent: if ``torch.ops.recstore_ops`` is already registered
(e.g. by a prior ``import recstore`` in the same process, or because another
library linked the same .so), we skip the second load and silence the
``OSError: already loaded`` exception.
"""

import glob
import os
import sys

import torch  # must be imported before the native extension


__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Native extension discovery
# ---------------------------------------------------------------------------
def _find_extension_so(pkg_path: str) -> str | None:
    """Locate the compiled ``_recstore_ops`` extension (.so).

    Order: wheel/install under recstore/lib/ -> in-place build under
    recstore/ -> legacy CMake build/lib/lib_recstore_ops.so -> setuptools
    build/lib.*/recstore/{lib/,}_recstore_ops.so.  Newest match wins when
    multiple candidates exist (typical during dev builds).
    """
    patterns = ("_recstore_ops*.so", "lib_recstore_ops*.so")
    repo_root = os.path.abspath(os.path.join(pkg_path, "..", "..", "..", ".."))

    def _first_under(dirs):
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for pat in patterns:
                matches = sorted(glob.glob(os.path.join(d, pat)))
                if matches:
                    # Prefer the newest (dev builds may leave stale copies).
                    matches.sort(key=os.path.getmtime, reverse=True)
                    return matches[0]
        return None

    # 1. Wheel install: recstore/lib/_recstore_ops.so
    # 2. In-place dev: recstore/_recstore_ops.so
    found = _first_under([
        os.path.join(pkg_path, "lib"),
        pkg_path,
    ])
    if found:
        return found

    # 3. Legacy CMake build: build/lib/lib_recstore_ops.so
    #    (covers users who ran `cmake --build build` but haven't adopted
    #    pip install yet)
    legacy = _first_under([os.path.join(repo_root, "build", "lib")])
    if legacy:
        return legacy

    # 4. setuptools build/lib.*/recstore/{lib/,}_recstore_ops.so
    #    (covers `python setup.py build` without --inplace)
    build_matches = []
    for pat in patterns:
        build_matches.extend(glob.glob(
            os.path.join(repo_root, "build", "lib.*", "recstore", "lib", pat)
        ))
        build_matches.extend(glob.glob(
            os.path.join(repo_root, "build", "lib.*", "recstore", pat)
        ))
    if build_matches:
        build_matches.sort(key=os.path.getmtime, reverse=True)
        return build_matches[0]

    return None


def _native_ops_ready() -> bool:
    """True if ``torch.ops.recstore_ops`` is already registered.

    Avoids a redundant ``load_library`` call when the extension was already
    loaded (e.g. by a sibling package or a second ``import recstore``).
    """
    try:
        ops_ns = getattr(torch.ops, "recstore_ops", None)
        if ops_ns is None:
            return False
        # torch.ops.<ns> is a OpNamespaceHolder; probing a known op is the
        # reliable way to confirm registration.  emb_read is in every build.
        return callable(getattr(ops_ns, "emb_read", None))
    except Exception:
        return False


def _ensure_native_loaded() -> str | None:
    """Idempotently load the recstore native extension.

    Returns the .so path on success, or ``None`` if the extension was
    already loaded.  Raises ``OSError`` if the .so cannot be found or
    loaded.
    """
    if _native_ops_ready():
        return None

    pkg_path = os.path.dirname(os.path.realpath(__file__))
    lib_path = _find_extension_so(pkg_path)
    if not lib_path or not os.path.isfile(lib_path):
        raise OSError(
            "RecStore: native extension not found.  Looked under "
            f"{pkg_path!r}/lib, {pkg_path!r}, the repo's build/lib/, and "
            "build/lib.*/recstore/{lib/,}.  Build the extension first via "
            "`python setup.py build_ext --inplace` (or `pip install -e .`), "
            "or run the CMake build (`cmake --build build`) which produces "
            "build/lib/lib_recstore_ops.so."
        )

    try:
        # torch.classes.load_library is the modern, idempotent-aware API
        # (PyTorch >= 1.13).  It also registers custom classes, not just ops.
        torch.classes.load_library(lib_path)
    except OSError as e:
        msg = str(e).lower()
        if "already loaded" in msg or "already initialized" in msg:
            # Another part of the process already loaded the same .so; this
            # is fine, the ops are registered.
            pass
        else:
            raise
    return lib_path


def load_ops_library() -> None:
    """Load the native extension (backwards-compatible alias)."""
    _ensure_native_loaded()


# Load the extension on import, unless explicitly deferred (for testing).
if os.environ.get("RECSTORE_DEFER_OPS_LOAD") != "1":
    _loaded_so_path = _ensure_native_loaded()
    if _loaded_so_path:
        if os.environ.get("RECSTORE_DEBUG", "") in ("1", "true", "TRUE"):
            print(f"[recstore] loaded native extension: {_loaded_so_path}")


# ---------------------------------------------------------------------------
# Build info (generated at build time by setup.py BuildPyCommand)
# ---------------------------------------------------------------------------
try:
    from . import version_info
    __build_info__ = version_info.get_version_info()
except ImportError:
    __build_info__ = {
        "version": __version__,
        "git": {
            "branch": "unknown",
            "commit_hash": "unknown",
            "commit_hash_full": "unknown",
            "commit_time": "unknown",
            "commit_author": "unknown",
            "commit_message": "unknown",
            "tag": "unknown",
        },
        "build": {
            "build_time": "unknown",
            "build_timestamp": 0,
            "python_version": "{}.{}.{}".format(*sys.version_info[:3]),
            "platform": sys.platform,
            "hostname": "unknown",
            "build_user": "unknown",
            "cuda_version": "",
            "cxx_abi": 1,
        },
    }


def get_build_info():
    """Return the build-time metadata dict (git commit, host, etc.)."""
    return __build_info__


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
    "get_build_info",
    "__version__",
    "__build_info__",
]
