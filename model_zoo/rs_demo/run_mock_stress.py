#!/usr/bin/env python3
"""Backward-compatible entrypoint."""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = str(_THIS_DIR.parents[1])
_PKG_PARENT = str(_THIS_DIR.parent)
# The bagpipe_cache runtime shim imports from ``python.pytorch.recstore.*``
# (the standalone package under src/).  inject_project_paths() adds src/ inside
# each worker, but the launcher imports the runner before any worker runs, so
# src/ must also be on the path here.
_RECSTORE_SRC = str(Path(_REPO_ROOT) / "src")
for _path in (_REPO_ROOT, _PKG_PARENT, _RECSTORE_SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rs_demo.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
