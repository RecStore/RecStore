from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import torch

from tools.config.recstore_config_path import resolve_recstore_config_path
from ..KVClient import RecStoreClient


class TestGpuCacheHitOnlyLookup(unittest.TestCase):
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

    def tearDown(self) -> None:
        try:
            self.client.disable_gpu_cache()
        finally:
            RecStoreClient._instance = None

    def test_hit_only_lookup_returns_cached_rows(self) -> None:
        table_name = f"gpu_cache_hit_only_{time.time_ns()}"
        self.client.init_data(
            name=table_name, shape=(64, 4), dtype=torch.float32
        )
        ids = torch.tensor([1, 3, 5], dtype=torch.int64, device="cuda")

        expected = self.client.local_lookup_flat(table_name, ids)
        actual = self.client.gpu_cache_lookup_flat_assuming_hits(ids, 4)

        self.assertTrue(torch.equal(actual, expected))

    def test_no_evict_prefill_reports_authoritative_residency(self) -> None:
        table_name = f"gpu_cache_no_evict_{time.time_ns()}"
        self.client.init_data(
            name=table_name, shape=(256, 4), dtype=torch.float32
        )
        keys = torch.arange(80, dtype=torch.int64, device="cuda")
        values = torch.arange(320, dtype=torch.float32, device="cuda").view(80, 4)

        inserted = self.client.prefill_gpu_cache_no_evict(
            table_name, keys, values
        )
        resident = self.client.contains_gpu_cache(keys)

        self.assertTrue(torch.equal(inserted, resident))
        self.assertTrue(bool(inserted.any().item()))
        self.assertTrue(
            torch.equal(
                self.client.gpu_cache_lookup_flat_assuming_hits(
                    keys[inserted], 4
                ),
                values[inserted],
            )
        )

        # A no-evict insert must never remove an already resident row.
        old_keys = keys[inserted]
        old_values = values[inserted]
        more_keys = torch.arange(
            100, 140, dtype=torch.int64, device="cuda"
        )
        more_values = torch.arange(
            160, dtype=torch.float32, device="cuda"
        ).view(40, 4)
        self.client.prefill_gpu_cache_no_evict(
            table_name, more_keys, more_values
        )
        self.assertTrue(
            torch.equal(
                self.client.gpu_cache_lookup_flat_assuming_hits(old_keys, 4),
                old_values,
            )
        )

    def test_no_evict_lookup_returns_values_and_residency_mask(self) -> None:
        table_name = f"gpu_cache_no_evict_lookup_{time.time_ns()}"
        self.client.init_data(
            name=table_name, shape=(256, 4), dtype=torch.float32
        )
        keys = torch.arange(80, dtype=torch.int64, device="cuda")
        expected = self.client.local_lookup_flat(table_name, keys)

        values, resident = self.client.gpu_cache_lookup_flat_no_evict(
            keys, 4
        )

        self.assertTrue(torch.equal(values, expected))
        self.assertTrue(torch.equal(resident, self.client.contains_gpu_cache(keys)))
        self.assertTrue(bool(resident.any().item()))

        more_keys = torch.arange(
            100, 140, dtype=torch.int64, device="cuda"
        )
        self.client.gpu_cache_lookup_flat_no_evict(more_keys, 4)
        old_values = self.client.gpu_cache_lookup_flat_assuming_hits(
            keys[resident], 4
        )
        self.assertTrue(torch.equal(old_values, expected[resident]))

    def test_gpu_cache_generation_changes_on_clear(self) -> None:
        before = self.client.get_gpu_cache_generation()
        self.client.clear_gpu_cache()
        after = self.client.get_gpu_cache_generation()
        self.assertGreater(after, before)

    def test_invalidate_with_mask_reports_authoritative_removal(self) -> None:
        table_name = f"gpu_cache_invalidate_mask_{time.time_ns()}"
        self.client.init_data(
            name=table_name, shape=(256, 4), dtype=torch.float32
        )
        keys = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device="cuda")
        values = torch.arange(16, dtype=torch.float32, device="cuda").view(4, 4)
        self.client.prefill_gpu_cache_no_evict(table_name, keys, values)

        removed = self.client.invalidate_gpu_cache_with_mask(
            table_name, keys[[1, 3]]
        )
        resident = self.client.contains_gpu_cache(keys)

        self.assertTrue(torch.equal(removed, torch.ones_like(removed)))
        self.assertEqual(
            resident.tolist(), [True, False, True, False]
        )
        again = self.client.invalidate_gpu_cache_with_mask(
            table_name, keys[[1, 3]]
        )
        self.assertFalse(bool(again.any().item()))


if __name__ == "__main__":
    unittest.main()
