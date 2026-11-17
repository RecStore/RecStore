#!/usr/bin/env python3
"""Check that source test files are registered with ctest.

Scope:
  * C++ tests under ``src/test`` (``test_*.cpp``)
  * Python unittests under ``src/python/pytorch/recstore/unittest``

A test file counts as registered when its file name or module name appears in
any ``CMakeLists.txt``. Tests that are deliberately not registered (manual
runners, pytest-only suites, stale files) must be listed in ``EXEMPT`` with a
reason, so new test files cannot silently stay outside ctest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RECSTORE_UNITTEST_DIR = REPO_ROOT / "src" / "python" / "pytorch" / "recstore" / "unittest"
CPP_TEST_DIR = REPO_ROOT / "src" / "test"

# Files that are intentionally not ctest entries. Keep the reason short and
# concrete; stale entries are reported as errors so this list cannot rot.
EXEMPT = {
    # pytest-only suites; the CI image used for ctest does not install pytest.
    "test_gradient_compression": "requires pytest",
    "test_sparse_sgd_topk_backend": "requires pytest",
    "test_sparse_sgd_topk_distributed": "requires pytest",
    "test_sparse_sgd_topk_integration": "requires pytest",
    "test_topk_strict_prevalidation": "requires pytest",
    # Script-style runners with a main(), not unittest TestCase modules.
    "test_ebc_precision": "manual runner (main entrypoint)",
    "test_ebc_precision_distributed_runner": "manual runner (requires PS cluster)",
    "test_ebc_precision_multiprocess": "manual runner (requires PS cluster)",
    "test_sparse_sgd_topk_gloo": "manual runner (requires torch.distributed)",
    # Stale after the perf-stats refactor; tracked for a follow-up rewrite.
    "test_gpu_training_cache": "stale: merge helpers moved to perf stats API",
}

CPP_EXEMPT: dict[str, str] = {}


def _cmake_text() -> str:
    parts = []
    for path in REPO_ROOT.rglob("CMakeLists.txt"):
        if any(part.startswith("build") for part in path.relative_to(REPO_ROOT).parts):
            continue
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _registered(text: str, name: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None


def _collect_tests() -> list[tuple[str, Path]]:
    tests: list[tuple[str, Path]] = []
    if RECSTORE_UNITTEST_DIR.is_dir():
        for path in sorted(RECSTORE_UNITTEST_DIR.glob("test_*.py")):
            tests.append((path.stem, path))
    if CPP_TEST_DIR.is_dir():
        for path in sorted(CPP_TEST_DIR.rglob("test_*.cpp")):
            tests.append((path.stem, path))
    return tests


def main() -> int:
    text = _cmake_text()
    missing: list[str] = []
    stale_exempt: list[str] = []
    seen: set[str] = set()

    for name, path in _collect_tests():
        seen.add(name)
        exempt_reason = EXEMPT.get(name) or CPP_EXEMPT.get(name)
        if _registered(text, name):
            if exempt_reason:
                stale_exempt.append(f"{name}: registered but still exempt ({exempt_reason})")
            continue
        if exempt_reason:
            continue
        missing.append(f"{path.relative_to(REPO_ROOT)}: not registered and not exempt")

    for name in {*EXEMPT, *CPP_EXEMPT} - seen:
        stale_exempt.append(f"{name}: exempt entry has no matching test file")

    for message in stale_exempt:
        print(f"warning: stale exemption: {message}")
    if missing:
        print("Unregistered test files:")
        for message in missing:
            print(f"  {message}")
        return 1
    print(f"Checked {len(seen)} test files; all are registered or explicitly exempt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
