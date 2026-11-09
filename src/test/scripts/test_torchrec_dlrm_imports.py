#!/usr/bin/env python3
"""Import smoke tests for the torchrec_dlrm model zoo entrypoints.

The model zoo used to import a non-existent ``python.pytorch.torchrec`` module.
These tests keep the import path working and catch the same drift in source.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
TORCHREC_DLRM = REPO_ROOT / "model_zoo" / "torchrec_dlrm"


def _ensure_src_on_path() -> None:
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTorchRecDlrmImports(unittest.TestCase):
    def test_torchrec_kv_embeddingbag_is_importable(self) -> None:
        _ensure_src_on_path()
        from python.pytorch.torchrec_kv.EmbeddingBag import (
            RecStoreEmbeddingBagCollection,
        )

        self.assertTrue(callable(RecStoreEmbeddingBagCollection))

    def test_dlrm_main_imports_with_torchrec_available(self) -> None:
        try:
            import torchrec  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest(f"torchrec_dlrm dependencies unavailable: {error}")

        # dlrm_main imports sibling modules (dlrm, data, lr_scheduler) by name.
        path = str(TORCHREC_DLRM)
        if path not in sys.path:
            sys.path.insert(0, path)
        _ensure_src_on_path()
        self.assertIsNotNone(_load_module(TORCHREC_DLRM / "dlrm_main.py"))

    def test_no_source_imports_the_removed_torchrec_package(self) -> None:
        stale = []
        for path in sorted(TORCHREC_DLRM.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "python.pytorch.torchrec." in text:
                stale.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(stale, [], "use python.pytorch.torchrec_kv instead")


if __name__ == "__main__":
    unittest.main()
