from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import torch

from tools.config.recstore_config_path import resolve_recstore_config_path
from ..KVClient import RecStoreClient


def _table(prefix: str) -> str:
    return f"{prefix}_{time.time_ns()}"


class TestGpuCacheInvalidation(unittest.TestCase):
    """Cache-coherence paths that the hit-only suite does not exercise.

    The GPU cache used to be observed through backend profile counters. That
    collection is gone, so every case here asserts residency
    (``contains_gpu_cache``), the generation or the clear counter, and the
    values the following read returns.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.library_path = (
            Path(__file__).resolve().parents[5] / "build/lib/lib_recstore_ops.so"
        )
        cls.config_path = resolve_recstore_config_path()
        if not cls.library_path.exists():
            raise unittest.SkipTest(f"missing ops library: {cls.library_path}")
        if not cls.config_path.exists():
            raise unittest.SkipTest(f"missing config file: {cls.config_path}")

    def setUp(self) -> None:
        os.environ["RECSTORE_CONFIG"] = str(self.config_path)
        RecStoreClient._instance = None
        self.client = RecStoreClient(str(self.library_path))
        self.client.set_ps_backend("hierkv")
        if not self.client.enable_gpu_cache(capacity=128, embedding_dim=4):
            raise unittest.SkipTest("GPU cache ops are not enabled")
        self.table = _table("gpu_cache_coherence")
        self.client.init_data(name=self.table, shape=(256, 4), dtype=torch.float32)

    def tearDown(self) -> None:
        os.environ.pop("RECSTORE_VALIDATE_GPU_CACHE_KEYS", None)
        try:
            self.client.disable_gpu_cache()
        finally:
            RecStoreClient._instance = None

    def _residency(self, ids: torch.Tensor) -> list[bool]:
        return self.client.contains_gpu_cache(ids).tolist()

    def _prefill(self, ids: torch.Tensor, fill: float) -> torch.Tensor:
        values = torch.full((ids.numel(), 4), fill, dtype=torch.float32, device="cuda")
        inserted = self.client.prefill_gpu_cache_no_evict(self.table, ids, values)
        self.assertTrue(bool(inserted.all().item()))
        return values

    def test_local_update_invalidates_then_refills(self) -> None:
        ids = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        stored = torch.full((2, 4), 5.0, dtype=torch.float32, device="cuda")
        self.client.push(self.table, ids, stored)
        self._prefill(ids, 5.0)
        self.assertEqual(self._residency(ids), [True, True])

        grads = torch.ones((2, 4), dtype=torch.float32, device="cuda")
        self.client.local_update_flat(self.table, ids, grads, learning_rate=0.0)
        self.assertEqual(self._residency(ids), [False, False])

        values = self.client.local_lookup_flat(self.table, ids)
        # A zero learning rate means "use the default", so the update applies
        # the default step rather than leaving the row untouched.
        self.assertTrue(torch.allclose(values, stored - 0.01))
        self.assertEqual(self._residency(ids), [True, True])
        # A refilled row answers the all-hit path again.
        _, miss = self.client.gpu_cache_lookup_flat_assuming_hits(ids, 4)
        self.assertFalse(bool(miss.any().item()))

    def test_cpu_id_update_clears_the_cache(self) -> None:
        ids = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        self._prefill(ids, 2.0)
        before = self.client.get_gpu_cache_clear_count()

        cpu_ids = torch.tensor([1, 3], dtype=torch.int64)
        grads = torch.ones((2, 4), dtype=torch.float32)
        self.client.local_update_flat(self.table, cpu_ids, grads, learning_rate=0.0)

        self.assertEqual(self._residency(ids), [False, False])
        self.assertEqual(self.client.get_gpu_cache_clear_count(), before + 1)

    def test_push_clears_the_cache(self) -> None:
        ids = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        self._prefill(ids, 1.0)
        before = self.client.get_gpu_cache_clear_count()

        pushed = torch.full((2, 4), 9.0, dtype=torch.float32, device="cuda")
        self.client.push(self.table, ids, pushed)

        self.assertEqual(self._residency(ids), [False, False])
        self.assertEqual(self.client.get_gpu_cache_clear_count(), before + 1)
        self.assertTrue(
            torch.equal(self.client.local_lookup_flat(self.table, ids), pushed)
        )

    def test_switching_tables_clears_the_cache(self) -> None:
        ids = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        self._prefill(ids, 3.0)
        other = _table("gpu_cache_other_table")
        self.client.init_data(name=other, shape=(256, 4), dtype=torch.float32)

        grads = torch.ones((2, 4), dtype=torch.float32, device="cuda")
        self.client.local_update_flat(other, ids, grads, learning_rate=0.0)

        self.assertEqual(self._residency(ids), [False, False])

    def test_low_hit_lookups_bypass_the_cache(self) -> None:
        self.assertTrue(self.client.is_gpu_cache_lookup_bypass_enabled())
        cached = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        self._prefill(cached, 4.0)
        keys = torch.arange(2048, 2048 + 2048, dtype=torch.int64, device="cuda")

        # The first all-miss lookup records the low hit streak; the second one
        # crosses the threshold and turns the cache off for large reads.
        self.client.local_lookup_flat(self.table, keys)
        self.assertFalse(self.client.is_gpu_cache_lookup_bypassed())
        self.client.local_lookup_flat(self.table, keys)

        self.assertTrue(self.client.is_gpu_cache_lookup_bypassed())
        # Two 2048-row fills overflow the 128-row cache, so the warm rows are
        # gone regardless of the bypass policy.
        self.assertEqual(self._residency(cached), [False, False])

        self.client.set_gpu_cache_lookup_bypass_enabled(False)
        self.client.reset_gpu_cache_bypass_state()
        self.assertFalse(self.client.is_gpu_cache_lookup_bypassed())

    def test_reserved_sentinel_keys_are_rejected(self) -> None:
        os.environ["RECSTORE_VALIDATE_GPU_CACHE_KEYS"] = "1"
        empty_key = torch.iinfo(torch.int64).max
        for reserved in (empty_key, empty_key - 1):
            with self.subTest(key=int(reserved)):
                ids = torch.tensor([reserved], dtype=torch.int64, device="cuda")
                with self.assertRaisesRegex(RuntimeError, "reserved"):
                    self.client.local_lookup_flat(self.table, ids)


if __name__ == "__main__":
    unittest.main()