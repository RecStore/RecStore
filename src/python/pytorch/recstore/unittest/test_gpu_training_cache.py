from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import torch

from model_zoo.rs_demo.runners.recstore_runner import (
    GPU_CACHE_PROFILE_KEYS,
    _add_sparse_id_stats,
    _merge_gpu_cache_profile,
)
from model_zoo.rs_demo.data.dlrm_source import build_sparse_features
from model_zoo.rs_demo.runtime.recstore_distributed import ShardedRecstoreClient
from tools.config.recstore_config_path import resolve_recstore_config_path
from ..KVClient import RecStoreClient


class _FakeGpuProfileOps:
    def __init__(self, values):
        self._values = values

    def get_last_gpu_cache_profile(self):
        return self._values


class _FakeGpuProfileClient:
    def __init__(self, values):
        self.ops = _FakeGpuProfileOps(values)


class _FakeShardOps:
    def __init__(self):
        self.backend = "BRPC"
        self.config_calls = []

    def current_ps_backend(self):
        return self.backend

    def set_ps_config(self, host, port):
        self.config_calls.append((host, int(port)))


class _FakeShardClient:
    def __init__(self):
        self.ops = _FakeShardOps()
        self.emb_read_calls = []

    def emb_read(self, keys: torch.Tensor, embedding_dim: int):
        self.emb_read_calls.append((keys.clone(), int(embedding_dim)))
        return torch.zeros((keys.numel(), int(embedding_dim)), dtype=torch.float32, device=keys.device)


class TestGpuCacheProfileMapping(unittest.TestCase):
    def tearDown(self) -> None:
        RecStoreClient._instance = None

    def _new_uninitialized_client(self, values) -> RecStoreClient:
        RecStoreClient._instance = None
        client = RecStoreClient.__new__(RecStoreClient)
        client.ops = _FakeGpuProfileOps(values)
        return client

    def test_recstore_client_parses_extended_gpu_cache_profile(self) -> None:
        client = self._new_uninitialized_client(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )

        self.assertEqual(
            client.get_last_gpu_cache_profile(),
            {
                "gpu_cache_query_ms": 1.0,
                "gpu_cache_backend_lookup_ms": 2.0,
                "gpu_cache_fill_ms": 3.0,
                "gpu_cache_update_ms": 4.0,
                "gpu_cache_hit_count": 5.0,
                "gpu_cache_invalidate_ms": 6.0,
                "gpu_cache_request_count": 7.0,
                "gpu_cache_miss_count": 8.0,
            },
        )

    def test_recstore_client_profile_keeps_old_profile_compatible(self) -> None:
        client = self._new_uninitialized_client([1.0, 2.0, 3.0, 4.0, 5.0])

        profile = client.get_last_gpu_cache_profile()

        self.assertEqual(profile["gpu_cache_invalidate_ms"], 0.0)
        self.assertEqual(profile["gpu_cache_request_count"], 0.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 0.0)

    def test_sharded_client_parses_extended_gpu_cache_profile_from_ops(self) -> None:
        client = object.__new__(ShardedRecstoreClient)
        client._client = _FakeGpuProfileClient([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        profile = client.get_last_gpu_cache_profile()

        self.assertEqual(profile["gpu_cache_invalidate_ms"], 6.0)
        self.assertEqual(profile["gpu_cache_request_count"], 7.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 8.0)

    def test_rs_demo_merges_extended_gpu_cache_profile_fields(self) -> None:
        row = {}
        kv_client = self._new_uninitialized_client(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )

        _merge_gpu_cache_profile(row, kv_client, "update")

        self.assertIn("gpu_cache_invalidate_ms", GPU_CACHE_PROFILE_KEYS)
        self.assertEqual(row["update_gpu_cache_invalidate_ms"], 6.0)
        self.assertEqual(row["update_gpu_cache_request_count"], 7.0)
        self.assertEqual(row["update_gpu_cache_miss_count"], 8.0)
        self.assertEqual(row["update_gpu_cache_hit_rate"], 5.0 / 7.0)

    def test_rs_demo_adds_sparse_id_scale_fields(self) -> None:
        features = build_sparse_features(
            ["f1", "f2"],
            torch.tensor([1, 1, 2, 3, 3, 4], dtype=torch.int64),
            torch.tensor([2, 1, 1, 2], dtype=torch.int32),
        )
        row = {}

        _add_sparse_id_stats(
            row,
            features,
            {"f1": 0, "f2": 1 << 30},
            cache_capacity=1024,
            prefetch_depth=2,
        )

        self.assertEqual(row["batch_raw_ids"], 6)
        self.assertEqual(row["batch_unique_ids"], 4)
        self.assertEqual(row["batch_dedup_ratio"], 2.0 / 6.0)
        self.assertEqual(row["gpu_cache_capacity"], 1024)
        self.assertEqual(row["prefetch_depth"], 2)

    def test_sharded_client_pull_with_gpu_cache_forwards_single_shard_ids(self) -> None:
        client = object.__new__(ShardedRecstoreClient)
        raw_client = _FakeShardClient()
        client._client = raw_client
        client._tensor_meta = {"table_a": {"shape": (16, 4), "dtype": torch.float32}}
        client._gpu_cache_table_name = None
        client._num_shards = 1
        client._cache_ps_type = "BRPC"
        client._cache_num_shards = 1
        client._servers = [type("Server", (), {"host": "127.0.0.1", "port": 15000, "shard": 0})()]
        client._cache_servers = client._servers
        client._servers_by_shard = {0: client._servers[0]}
        client._active_shard = None
        client._native_distributed_backend = False

        ids = torch.tensor([1, 3], dtype=torch.int64)
        client.pull_with_gpu_cache("table_a", ids)

        self.assertEqual(len(raw_client.emb_read_calls), 1)
        called_ids, called_dim = raw_client.emb_read_calls[0]
        self.assertTrue(torch.equal(called_ids, ids))
        self.assertEqual(called_dim, 4)
        self.assertEqual(raw_client.ops.config_calls, [("127.0.0.1", 15000)])


class TestGpuTrainingCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.repo_root = Path(__file__).resolve().parents[5]
        cls.library_path = cls.repo_root / "build/lib/lib_recstore_ops.so"
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

    def _new_table_name(self) -> str:
        return f"gpu_cache_it_{time.time_ns()}"

    def test_lookup_miss_fills_cache_and_second_lookup_hits(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([1, 3, 5], dtype=torch.int64, device="cuda")

        first = self.client.local_lookup_flat(table_name, ids)
        first_profile = self.client.get_last_gpu_cache_profile()
        second = self.client.local_lookup_flat(table_name, ids)
        second_profile = self.client.get_last_gpu_cache_profile()

        self.assertEqual(first.device.type, "cuda")
        self.assertTrue(torch.allclose(first, torch.zeros((3, 4), device="cuda")))
        self.assertTrue(torch.allclose(second, first))
        self.assertEqual(first_profile["gpu_cache_hit_count"], 0.0)
        self.assertEqual(second_profile["gpu_cache_hit_count"], 3.0)

    def test_update_invalidates_cached_rows_before_next_lookup_refills(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([7, 9], dtype=torch.int64, device="cuda")

        before = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(before, torch.zeros((2, 4), device="cuda")))
        cached = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(cached, before))
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

        grads = torch.ones((2, 4), dtype=torch.float32, device="cuda")
        self.client.local_update_flat(table_name, ids, grads)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))
        profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(profile["gpu_cache_hit_count"], 0.0)
        self.assertEqual(profile["gpu_cache_request_count"], 2.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 2.0)

        refilled = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(refilled, expected))
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

    def test_apply_sgd_update_gpu_cache_keeps_cached_rows_visible(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([7, 9], dtype=torch.int64, device="cuda")

        before = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(before, torch.zeros((2, 4), device="cuda")))
        cached = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(cached, before))
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

        grads = torch.ones((2, 4), dtype=torch.float32, device="cuda")
        self.assertTrue(
            self.client.apply_sgd_update_gpu_cache(
                table_name,
                ids,
                grads,
                learning_rate=0.01,
            )
        )
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))
        profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(profile["gpu_cache_hit_count"], 2.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 0.0)

    def test_invalidate_gpu_cache_evicts_rows_before_next_lookup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([11], dtype=torch.int64, device="cuda")

        before = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(before, torch.zeros((1, 4), device="cuda")))
        cached = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(cached, before))
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 1.0)

        grads = torch.ones((1, 4), dtype=torch.float32)
        self.client.local_update_flat(table_name, ids.cpu(), grads)
        self.client.invalidate_gpu_cache(table_name, ids)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((1, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))
        profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(profile["gpu_cache_hit_count"], 0.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 1.0)

    def test_update_invalidation_preserves_duplicate_id_updates(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([7, 7], dtype=torch.int64, device="cuda")

        self.client.local_lookup_flat(table_name, ids)
        self.client.local_lookup_flat(table_name, ids)
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

        grads = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.client.local_update_flat(table_name, ids, grads)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.03, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))
        profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(profile["gpu_cache_hit_count"], 0.0)
        self.assertEqual(profile["gpu_cache_request_count"], 2.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 2.0)

    def test_emb_update_table_invalidates_cached_rows_before_next_lookup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([21, 23], dtype=torch.int64, device="cuda")
        self.client.local_lookup_flat(table_name, ids)
        self.client.local_lookup_flat(table_name, ids)
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

        grads = torch.ones((2, 4), dtype=torch.float32)
        self.client.ops.emb_update_table(table_name, ids.cpu(), grads)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 0.0)

    def test_cpu_ids_update_clears_gpu_cache(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids_cuda = torch.tensor([17, 19], dtype=torch.int64, device="cuda")
        self.client.local_lookup_flat(table_name, ids_cuda)
        self.client.local_lookup_flat(table_name, ids_cuda)
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

        grads_cpu = torch.ones((2, 4), dtype=torch.float32)
        self.client.local_update_flat(table_name, ids_cuda.cpu(), grads_cpu)
        after = self.client.local_lookup_flat(table_name, ids_cuda)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))
        profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(profile["gpu_cache_hit_count"], 0.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 2.0)

    def test_push_invalidates_cached_rows_before_next_lookup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([11, 12], dtype=torch.int64, device="cuda")
        self.client.local_lookup_flat(table_name, ids)
        self.client.local_lookup_flat(table_name, ids)
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

        replacement = torch.full((2, 4), 3.0, dtype=torch.float32)
        self.client.push(table_name, ids.cpu(), replacement)
        out = self.client.local_lookup_flat(table_name, ids)

        self.assertTrue(torch.allclose(out, replacement.to("cuda")))
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 0.0)

    def test_switching_tables_invalidates_cache(self) -> None:
        table_a = self._new_table_name()
        table_b = self._new_table_name()
        self.client.init_data(name=table_a, shape=(64, 4), dtype=torch.float32)
        self.client.init_data(name=table_b, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([13, 14], dtype=torch.int64, device="cuda")
        self.client.push(table_b, ids.cpu(), torch.full((2, 4), 5.0, dtype=torch.float32))

        self.client.local_lookup_flat(table_a, ids)
        self.client.local_lookup_flat(table_a, ids)
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 2.0)

        out_b = self.client.local_lookup_flat(table_b, ids)

        self.assertTrue(torch.allclose(out_b, torch.full((2, 4), 5.0, device="cuda")))
        self.assertEqual(self.client.get_last_gpu_cache_profile()["gpu_cache_hit_count"], 0.0)

    def test_duplicate_ids_preserve_request_order_after_cache_fill(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        seed_ids = torch.tensor([4, 6, 8, 10], dtype=torch.int64, device="cuda")
        seed_grads = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.client.local_update_flat(table_name, seed_ids, seed_grads)
        self.client.clear_gpu_cache()

        self.client.local_lookup_flat(
            table_name, torch.tensor([4, 6], dtype=torch.int64, device="cuda")
        )

        ids = torch.tensor([4, 8, 6, 10, 8], dtype=torch.int64, device="cuda")
        out = self.client.local_lookup_flat(table_name, ids)
        expected = torch.tensor(
            [
                [-0.01, -0.01, -0.01, -0.01],
                [-0.03, -0.03, -0.03, -0.03],
                [-0.02, -0.02, -0.02, -0.02],
                [-0.04, -0.04, -0.04, -0.04],
                [-0.03, -0.03, -0.03, -0.03],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.assertTrue(torch.allclose(out, expected))
        profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(profile["gpu_cache_hit_count"], 2.0)

        cached = self.client.local_lookup_flat(table_name, ids)
        cached_expected = torch.tensor(
            [
                [-0.01, -0.01, -0.01, -0.01],
                [-0.03, -0.03, -0.03, -0.03],
                [-0.02, -0.02, -0.02, -0.02],
                [-0.04, -0.04, -0.04, -0.04],
                [-0.03, -0.03, -0.03, -0.03],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.assertTrue(torch.allclose(cached, cached_expected))

    def test_large_low_hit_lookups_bypass_gpu_cache_after_warmup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(20_000, 4), dtype=torch.float32)

        ids = torch.arange(0, 4096, dtype=torch.int64, device="cuda")
        out = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(out, torch.zeros((4096, 4), device="cuda")))
        profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(profile["gpu_cache_hit_count"], 0.0)
        self.assertEqual(profile["gpu_cache_request_count"], 4096.0)
        self.assertEqual(profile["gpu_cache_miss_count"], 4096.0)

        bypass_ids = torch.arange(4096, 4096 + 4096, dtype=torch.int64, device="cuda")
        bypassed = self.client.local_lookup_flat(table_name, bypass_ids)
        self.assertTrue(torch.allclose(bypassed, torch.zeros((4096, 4), device="cuda")))
        bypass_profile = self.client.get_last_gpu_cache_profile()
        self.assertEqual(bypass_profile["gpu_cache_request_count"], 0.0)
        self.assertEqual(bypass_profile["gpu_cache_miss_count"], 0.0)
        self.assertEqual(bypass_profile["gpu_cache_hit_count"], 0.0)

    def test_gpu_cache_rejects_reserved_sentinel_keys(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        reserved_ids = torch.tensor(
            [
                torch.iinfo(torch.int64).max,
                torch.iinfo(torch.int64).max - 1,
            ],
            dtype=torch.int64,
            device="cuda",
        )

        old_value = os.environ.get("RECSTORE_VALIDATE_GPU_CACHE_KEYS")
        os.environ["RECSTORE_VALIDATE_GPU_CACHE_KEYS"] = "1"
        try:
            with self.assertRaisesRegex(RuntimeError, "reserved GPU cache sentinel"):
                self.client.local_lookup_flat(table_name, reserved_ids)
        finally:
            if old_value is None:
                os.environ.pop("RECSTORE_VALIDATE_GPU_CACHE_KEYS", None)
            else:
                os.environ["RECSTORE_VALIDATE_GPU_CACHE_KEYS"] = old_value


if __name__ == "__main__":
    unittest.main()
