"""Python entry point for the C++ metric reporter.

``report()`` is exported by ``libreport.so``, the shared build of
``src/base/report/report_client.cpp``. Calling it through ctypes puts Python
samples into the same sinks as the C++ call sites (``remote`` / ``csv`` /
``jsonl`` / ``glog``), so no separate Python reporting path is needed.

The library only exists when the tree was configured with
``-DUSE_PERF_REPORT=ON``. When it is missing, :func:`available` returns
``False`` and :func:`emit` returns ``False`` instead of raising, so callers can
report unconditionally.
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

__all__ = ["available", "emit", "library_path", "report_metric", "reset"]

_LIBRARY_NAME = "libreport.so"
_lib: ctypes.CDLL | None = None
_lib_path: str | None = None
_loaded = False


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    # .../src/python/pytorch/recstore/telemetry.py -> repository root
    return _package_dir().parents[3]


def _candidate_paths() -> list[Path]:
    override = os.environ.get("RECSTORE_REPORT_LIBRARY")
    if override:
        return [Path(override)]

    package = _package_dir()
    candidates = [
        package / "lib" / _LIBRARY_NAME,
        package / _LIBRARY_NAME,
        _repo_root() / "build" / "lib" / _LIBRARY_NAME,
    ]
    # Build directories are not always named "build"; prefer the newest copy
    # so a stale tree does not win over a fresh one.
    discovered = glob.glob(
        str(_repo_root() / "build*" / "lib" / _LIBRARY_NAME)
    )
    discovered.sort(key=lambda item: os.path.getmtime(item), reverse=True)
    candidates.extend(Path(item) for item in discovered)
    return candidates


def _load() -> ctypes.CDLL | None:
    global _loaded, _lib, _lib_path
    if _loaded:
        return _lib
    _loaded = True

    for candidate in _candidate_paths():
        if not candidate.is_file():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError:
            continue
        library.report.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_double,
        ]
        library.report.restype = ctypes.c_bool
        _lib = library
        _lib_path = str(candidate)
        return library
    return None


def available() -> bool:
    """True when ``libreport.so`` was found and loaded."""
    return _load() is not None


def library_path() -> str | None:
    """Path of the loaded ``libreport.so``, or ``None``."""
    _load()
    return _lib_path


def emit(table: str, uid: str, metric: str, value: float) -> bool:
    """Report one sample. Returns False when reporting is not compiled in."""
    library = _load()
    if library is None:
        return False
    return bool(
        library.report(
            table.encode("utf-8"),
            uid.encode("utf-8"),
            metric.encode("utf-8"),
            float(value),
        )
    )


# Name used by the older call sites in KVClient; kept as an alias.
report_metric = emit


def reset() -> None:
    """Forget the loaded library. Tests use this after changing the search path."""
    global _loaded, _lib, _lib_path
    _loaded = False
    _lib = None
    _lib_path = None
