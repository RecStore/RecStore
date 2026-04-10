#!/usr/bin/env python3
"""Backward-compatible entrypoint."""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_PARENT = str(_THIS_DIR.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rs_demo.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
