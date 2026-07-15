from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from model_zoo.rs_demo.runtime.recstore_distributed import (
    ShardedRecstoreClient,
    _city_hash64_of_uint64,
)


class _FakeOps:
    def __init__(self) -> None:
        self.active_port: int | None = None
        self.port_history: list[int] = []
        self.backend = "brpc"
        self.lookup_calls: list[tuple[int | None, list[int], int]] = []
        self.update_calls: list[tuple[int | None, str, list[int], list[list[float]]]] = []
        self.update_clear_gpu_cache_call_counts: list[int] = []
        self.backend_switch_calls: list[str] = []
        self.lookup_region_warmup_calls = 0
        self.clear_gpu_cache_calls = 0

    def set_ps_config(self, host: str, port: int) -> None:
        self.active_port = int(port)
        self.port_history.append(int(port))

    def current_ps_backend(self) -> str:
        return self.backend

    def set_ps_backend(self, backend: str) -> None:
        self.backend = backend
        self.backend_switch_calls.append(str(backend))

    def local_lookup_flat(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.lookup_calls.append(
            (self.active_port, [int(v) for v in keys.tolist()], int(embedding_dim))
        )
        rows = keys.numel()
        return torch.arange(rows * int(embedding_dim), dtype=torch.float32).view(rows, int(embedding_dim))

    def local_update_flat(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        self.update_clear_gpu_cache_call_counts.append(self.clear_gpu_cache_calls)
        self.update_calls.append(
            (
                self.active_port,
                table_name,
                [int(v) for v in keys.tolist()],
                [[float(x) for x in row] for row in grads.tolist()],
            )
        )

    def warmup_local_lookup_flat_cuda_region(self) -> bool:
        self.lookup_region_warmup_calls += 1
        return True

    def clear_gpu_cache(self) -> None:
        self.clear_gpu_cache_calls += 1


class _FakeOpsWithGpuCache(_FakeOps):
    def __init__(self) -> None:
        super().__init__()
        self.gpu_cache_calls: list[tuple[int, int]] = []
        self.gpu_cache_sgd_update_calls: list[tuple[list[int], list[list[float]], float]] = []
        self.gpu_cache_sgd_update_devices: list[tuple[str, str]] = []
        self.gpu_cache_invalidate_calls: list[list[int]] = []
        self.gpu_cache_invalidate_devices: list[str] = []
        self.bypass_enabled = True

    def enable_gpu_cache(self, capacity: int, embedding_dim: int) -> bool:
        self.gpu_cache_calls.append((int(capacity), int(embedding_dim)))
        return True

    def set_gpu_cache_lookup_bypass_enabled(self, enabled: bool) -> None:
        self.bypass_enabled = bool(enabled)

    def is_gpu_cache_lookup_bypass_enabled(self) -> bool:
        return self.bypass_enabled

    def is_gpu_cache_lookup_bypassed(self) -> bool:
        return not self.bypass_enabled

    def reset_gpu_cache_bypass_state(self) -> None:
        self.bypass_enabled = True

    def apply_sgd_update_gpu_cache(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        learning_rate: float,
    ) -> bool:
        self.gpu_cache_sgd_update_devices.append((str(ids.device), str(grads.device)))
        self.gpu_cache_sgd_update_calls.append(
            (
                [int(v) for v in ids.tolist()],
                [[float(x) for x in row] for row in grads.tolist()],
                float(learning_rate),
            )
        )
        return True

    def invalidate_gpu_cache(self, ids: torch.Tensor) -> None:
        self.gpu_cache_invalidate_devices.append(str(ids.device))
        self.gpu_cache_invalidate_calls.append([int(v) for v in ids.tolist()])


class _FakeClient:
    def __init__(self) -> None:
        self.ops = _FakeOps()
        self.table_inits: list[tuple[int, str, int, int]] = []
        self.writes: dict[int, dict[int, list[float]]] = {}
        self.updates: list[tuple[int, str, list[int], list[list[float]]]] = []
        self.prefetch_requests: dict[int, tuple[int, list[int]]] = {}
        self._next_prefetch_id = 1

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        assert self.ops.active_port is not None
        self.table_inits.append(
            (self.ops.active_port, table_name, int(num_embeddings), int(embedding_dim))
        )
        return True

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        assert self.ops.active_port is not None
        bucket = self.writes.setdefault(self.ops.active_port, {})
        for key, value in zip(keys.tolist(), values.tolist()):
            bucket[int(key)] = [float(x) for x in value]

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        assert self.ops.active_port is not None
        rows = []
        bucket = self.writes.setdefault(self.ops.active_port, {})
        for key in keys.tolist():
            rows.append(bucket[int(key)])
        return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), embedding_dim)

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        assert self.ops.active_port is not None
        self.updates.append(
            (
                self.ops.active_port,
                table_name,
                [int(v) for v in keys.tolist()],
                [[float(x) for x in row] for row in grads.tolist()],
            )
        )

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        assert self.ops.active_port is not None
        pid = self._next_prefetch_id
        self._next_prefetch_id += 1
        self.prefetch_requests[pid] = (self.ops.active_port, [int(v) for v in keys.tolist()])
        return pid

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        port, keys = self.prefetch_requests[int(prefetch_id)]
        bucket = self.writes.setdefault(port, {})
        rows = [bucket[int(key)] for key in keys]
        return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), embedding_dim)


class _FakeClientCollidingPrefetchId(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.prefetch_requests_by_port: dict[tuple[int, int], list[int]] = {}

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        assert self.ops.active_port is not None
        # Simulate backend with per-connection/local handle namespace:
        # different shards may return the same id.
        self.prefetch_requests_by_port[(self.ops.active_port, 1)] = [
            int(v) for v in keys.tolist()
        ]
        return 1

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        assert self.ops.active_port is not None
        keys = self.prefetch_requests_by_port[(self.ops.active_port, int(prefetch_id))]
        bucket = self.writes.setdefault(self.ops.active_port, {})
        rows = [bucket[int(key)] for key in keys]
        return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), embedding_dim)


class _FakeClientWithGpuCache(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.gpu_cache_calls: list[tuple[int, int]] = []
        self.gpu_cache_profile = {
            "gpu_cache_query_ms": 1.0,
            "gpu_cache_backend_lookup_ms": 2.0,
            "gpu_cache_fill_ms": 3.0,
            "gpu_cache_update_ms": 4.0,
            "gpu_cache_hit_count": 5.0,
        }
        self.bypass_enabled = True

    def enable_gpu_cache(self, capacity: int, embedding_dim: int) -> bool:
        self.gpu_cache_calls.append((int(capacity), int(embedding_dim)))
        return True

    def get_last_gpu_cache_profile(self) -> dict[str, float]:
        return self.gpu_cache_profile

    def set_gpu_cache_lookup_bypass_enabled(self, enabled: bool) -> None:
        self.bypass_enabled = bool(enabled)

    def is_gpu_cache_lookup_bypass_enabled(self) -> bool:
        return self.bypass_enabled


class _FakeClientWithOpsGpuCache(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.ops = _FakeOpsWithGpuCache()


class _FakeClientWithClientAndOpsGpuCache(_FakeClientWithOpsGpuCache):
    def apply_sgd_update_gpu_cache(self, name, ids, grads, *, learning_rate):
        del name, ids, grads, learning_rate
        raise RuntimeError("raw client metadata is intentionally unavailable")


class _FakeClientWithWriteRowLimit(_FakeClient):
    def __init__(self, max_rows_per_write: int) -> None:
        super().__init__()
        self.max_rows_per_write = int(max_rows_per_write)
        self.write_batch_sizes: list[int] = []

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.write_batch_sizes.append(int(keys.numel()))
        if keys.numel() > self.max_rows_per_write:
            raise RuntimeError("single emb_write request exceeds fake slot limit")
        super().emb_write(keys, values)


class _FakeNativeDistributedClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.ops.backend = "distributed_grpc"
        self.direct_writes: dict[int, list[float]] = {}
        self.direct_updates: list[tuple[str, list[int], list[list[float]]]] = []
        self.direct_prefetch_requests: dict[int, list[int]] = {}

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        self.table_inits.append((-1, table_name, int(num_embeddings), int(embedding_dim)))
        return True

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        for key, value in zip(keys.tolist(), values.tolist()):
            self.direct_writes[int(key)] = [float(x) for x in value]

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        rows = [self.direct_writes[int(key)] for key in keys.tolist()]
        return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), embedding_dim)

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        self.direct_updates.append(
            (
                table_name,
                [int(v) for v in keys.tolist()],
                [[float(x) for x in row] for row in grads.tolist()],
            )
        )

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        pid = self._next_prefetch_id
        self._next_prefetch_id += 1
        self.direct_prefetch_requests[pid] = [int(v) for v in keys.tolist()]
        return pid

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        keys = self.direct_prefetch_requests[int(prefetch_id)]
        rows = [self.direct_writes[int(key)] for key in keys]
        return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), embedding_dim)


class TestShardedRecstoreClient(unittest.TestCase):
    def _make_runtime_dir(
        self,
        *,
        hash_method: str = "simple_mod",
        cache_ps_type: str | None = None,
        distributed_servers: list[dict] | None = None,
        cache_servers: list[dict] | None = None,
        distributed_num_shards: int | None = None,
        max_keys_per_request: int | None = None,
        include_distributed_servers: bool = True,
        include_distributed_num_shards: bool = True,
    ) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        runtime_dir = Path(tmpdir.name)
        if cache_servers is None:
            cache_servers = [
                {"host": "127.0.0.1", "port": 20000, "shard": 0},
                {"host": "127.0.0.1", "port": 20001, "shard": 1},
            ]
        if distributed_servers is None:
            distributed_servers = [
                {"host": "127.0.0.1", "port": 20000, "shard": 0},
                {"host": "127.0.0.1", "port": 20001, "shard": 1},
            ]
        if distributed_num_shards is None:
            distributed_num_shards = len(distributed_servers)
        cfg = {
            "cache_ps": {
                "servers": cache_servers,
            },
            "distributed_client": {
                "hash_method": hash_method,
            },
        }
        if cache_ps_type is not None:
            cfg["cache_ps"]["ps_type"] = cache_ps_type
        if include_distributed_num_shards:
            cfg["distributed_client"]["num_shards"] = distributed_num_shards
        if max_keys_per_request is not None:
            cfg["distributed_client"]["max_keys_per_request"] = int(max_keys_per_request)
        if include_distributed_servers:
            cfg["distributed_client"]["servers"] = distributed_servers
        (runtime_dir / "recstore_config.json").write_text(
            json.dumps(cfg),
            encoding="utf-8",
        )
        return runtime_dir

    @staticmethod
    def _cityhash_shard_for_key(key: int, num_shards: int) -> int:
        return int(_city_hash64_of_uint64(int(key)) % num_shards)

    def test_routes_init_write_read_and_update_by_shard(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        self.assertTrue(client.init_embedding_table("default", 100, 4))

        keys = torch.tensor([5, 2, 7, 4], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        read_back = client.emb_read(keys, 4)
        self.assertTrue(torch.allclose(read_back, values))

        grads = values + 100.0
        client.emb_update_table("default", keys, grads)

        self.assertEqual(
            fake_client.table_inits,
            [
                (20000, "default", 100, 4),
                (20001, "default", 100, 4),
            ],
        )
        self.assertEqual(sorted(fake_client.writes[20000].keys()), [2, 4])
        self.assertEqual(sorted(fake_client.writes[20001].keys()), [5, 7])
        self.assertEqual(
            fake_client.updates,
            [
                (20001, "default", [5, 7], grads[[0, 2]].tolist()),
                (20000, "default", [2, 4], grads[[1, 3]].tolist()),
            ],
        )

    def test_city_hash_routing_matches_backend_semantics(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="city_hash")
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([2, 4, 5, 7], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        expected_port_to_keys: dict[int, list[int]] = {20000: [], 20001: []}
        for key in keys.tolist():
            shard = self._cityhash_shard_for_key(int(key), 2)
            port = 20000 if shard == 0 else 20001
            expected_port_to_keys[port].append(int(key))

        self.assertEqual(sorted(fake_client.writes[20000].keys()), sorted(expected_port_to_keys[20000]))
        self.assertEqual(sorted(fake_client.writes[20001].keys()), sorted(expected_port_to_keys[20001]))

    def test_prefers_distributed_client_servers_over_cache_ps(self) -> None:
        runtime_dir = self._make_runtime_dir(
            hash_method="simple_mod",
            distributed_servers=[
                {"host": "127.0.0.1", "port": 21000, "shard": 0},
                {"host": "127.0.0.1", "port": 21001, "shard": 1},
            ],
            cache_servers=[
                {"host": "127.0.0.1", "port": 22000, "shard": 0},
                {"host": "127.0.0.1", "port": 22001, "shard": 1},
            ],
            distributed_num_shards=2,
        )
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([0, 1], dtype=torch.int64)
        values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        client.emb_write(keys, values)

        self.assertIn(21000, fake_client.writes)
        self.assertIn(21001, fake_client.writes)
        self.assertNotIn(22000, fake_client.writes)
        self.assertNotIn(22001, fake_client.writes)

    def test_native_distributed_backend_does_not_switch_shards(self) -> None:
        runtime_dir = self._make_runtime_dir(
            hash_method="simple_mod",
            cache_ps_type="GRPC",
            distributed_num_shards=2,
        )
        fake_client = _FakeNativeDistributedClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        self.assertTrue(client.init_embedding_table("default", 100, 4))
        keys = torch.tensor([5, 2, 7, 4], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        handle = client.prefetch(keys)
        self.assertEqual(fake_client.ops.port_history, [])
        self.assertEqual(fake_client.direct_prefetch_requests[handle], [5, 2, 7, 4])
        self.assertTrue(torch.allclose(client.wait_and_get(handle, 4), values))
        self.assertEqual(fake_client.table_inits, [(-1, "default", 100, 4)])

    def test_routes_by_shard_id_when_shards_are_non_contiguous(self) -> None:
        runtime_dir = self._make_runtime_dir(
            hash_method="simple_mod",
            distributed_servers=[
                {"host": "127.0.0.1", "port": 23000, "shard": 0},
                {"host": "127.0.0.1", "port": 23002, "shard": 2},
            ],
            distributed_num_shards=3,
        )
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([0, 2, 5], dtype=torch.int64)  # mod 3 -> shard 0,2,2
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        client.emb_write(keys, values)

        self.assertEqual(sorted(fake_client.writes[23000].keys()), [0])
        self.assertEqual(sorted(fake_client.writes[23002].keys()), [2, 5])

    def test_unknown_hash_method_falls_back_to_city_hash(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="unknown_hash_name", distributed_num_shards=2)
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([2, 4, 5, 7], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        expected_port_to_keys: dict[int, list[int]] = {20000: [], 20001: []}
        for key in keys.tolist():
            shard = self._cityhash_shard_for_key(int(key), 2)
            port = 20000 if shard == 0 else 20001
            expected_port_to_keys[port].append(int(key))

        self.assertEqual(sorted(fake_client.writes[20000].keys()), sorted(expected_port_to_keys[20000]))
        self.assertEqual(sorted(fake_client.writes[20001].keys()), sorted(expected_port_to_keys[20001]))

    def test_cityhash_library_is_loaded_lazily(self) -> None:
        with mock.patch("ctypes.CDLL", side_effect=OSError("lib load boom")):
            module = importlib.import_module("model_zoo.rs_demo.runtime.recstore_distributed")
            importlib.reload(module)

    def test_fallback_to_cache_ps_servers_when_distributed_servers_missing(self) -> None:
        runtime_dir = self._make_runtime_dir(
            hash_method="simple_mod",
            cache_servers=[
                {"host": "127.0.0.1", "port": 24000, "shard": 0},
                {"host": "127.0.0.1", "port": 24001, "shard": 1},
            ],
            include_distributed_servers=False,
        )
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([0, 1], dtype=torch.int64)
        values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        client.emb_write(keys, values)

        self.assertIn(24000, fake_client.writes)
        self.assertIn(24001, fake_client.writes)

    def test_fallback_num_shards_to_cache_then_servers_len(self) -> None:
        runtime_dir_cache = self._make_runtime_dir(
            hash_method="simple_mod",
            distributed_servers=[
                {"host": "127.0.0.1", "port": 25000, "shard": 0},
                {"host": "127.0.0.1", "port": 25002, "shard": 2},
            ],
            distributed_num_shards=3,
            include_distributed_num_shards=False,
        )
        cfg_cache = json.loads((runtime_dir_cache / "recstore_config.json").read_text(encoding="utf-8"))
        cfg_cache["cache_ps"]["num_shards"] = 3
        (runtime_dir_cache / "recstore_config.json").write_text(json.dumps(cfg_cache), encoding="utf-8")
        fake_client_cache = _FakeClient()
        client_cache = ShardedRecstoreClient(fake_client_cache, runtime_dir_cache)
        keys = torch.tensor([0, 2, 5], dtype=torch.int64)  # mod 3 -> 0,2,2
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        client_cache.emb_write(keys, values)
        self.assertEqual(sorted(fake_client_cache.writes[25000].keys()), [0])
        self.assertEqual(sorted(fake_client_cache.writes[25002].keys()), [2, 5])

        runtime_dir_len = self._make_runtime_dir(
            hash_method="simple_mod",
            distributed_servers=[
                {"host": "127.0.0.1", "port": 25100, "shard": 0},
                {"host": "127.0.0.1", "port": 25101, "shard": 1},
            ],
            include_distributed_num_shards=False,
        )
        cfg_len = json.loads((runtime_dir_len / "recstore_config.json").read_text(encoding="utf-8"))
        cfg_len["cache_ps"].pop("num_shards", None)
        (runtime_dir_len / "recstore_config.json").write_text(json.dumps(cfg_len), encoding="utf-8")
        fake_client_len = _FakeClient()
        client_len = ShardedRecstoreClient(fake_client_len, runtime_dir_len)
        keys_len = torch.tensor([0, 1], dtype=torch.int64)
        values_len = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        client_len.emb_write(keys_len, values_len)
        self.assertIn(25100, fake_client_len.writes)
        self.assertIn(25101, fake_client_len.writes)

    def test_prefetch_wait_results_are_reassembled_in_input_order(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([5, 2, 7, 4], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        pid = client.emb_prefetch(keys)
        self.assertEqual(fake_client.prefetch_requests, {})
        out = client.emb_wait_result(pid, 4)
        self.assertTrue(torch.allclose(out, values))
        self.assertEqual(
            sorted((port, tuple(keys)) for port, keys in fake_client.prefetch_requests.values()),
            [(20000, (2, 4)), (20001, (5, 7))],
        )

    def test_stable_prefetch_read_handles_colliding_shard_prefetch_ids(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClientCollidingPrefetchId()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([5, 2, 7, 4], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        out = client.emb_read_prefetch(keys, 4)
        self.assertTrue(torch.allclose(out, values))

    def test_public_prefetch_wait_handles_colliding_shard_prefetch_ids(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClientCollidingPrefetchId()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([5, 2, 7, 4], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        opaque_handle = client.emb_prefetch(keys)
        self.assertEqual(fake_client.prefetch_requests_by_port, {})
        out = client.emb_wait_result(opaque_handle, 4)
        self.assertTrue(torch.allclose(out, values))
        self.assertEqual(
            sorted(fake_client.prefetch_requests_by_port.items()),
            [((20000, 1), [2, 4]), ((20001, 1), [5, 7])],
        )

    def test_public_prefetch_wait_consumes_opaque_handle(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([5, 2, 7, 4], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        opaque_handle = client.emb_prefetch(keys)
        out = client.emb_wait_result(opaque_handle, 4)
        self.assertTrue(torch.allclose(out, values))
        with self.assertRaises(RuntimeError):
            client.emb_wait_result(opaque_handle, 4)

    def test_init_data_and_pull_routes_to_shards(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        init_values = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        client.init_data(
            name="fused",
            shape=(6, 2),
            dtype=torch.float32,
            base_offset=100,
            init_func=lambda shape, dtype: init_values,
        )

        ids = torch.arange(100, 106, dtype=torch.int64)
        pulled = client.pull("fused", ids)
        self.assertTrue(torch.allclose(pulled, init_values))

        shard_to_keys = {20000: [], 20001: []}
        for key in ids.tolist():
            port = 20000 if key % 2 == 0 else 20001
            shard_to_keys[port].append(int(key))
        self.assertEqual(sorted(fake_client.writes[20000].keys()), sorted(shard_to_keys[20000]))
        self.assertEqual(sorted(fake_client.writes[20001].keys()), sorted(shard_to_keys[20001]))

    def test_init_data_splits_large_writes_to_fit_local_shm_slot(self) -> None:
        runtime_dir = self._make_runtime_dir(
            cache_ps_type="LOCAL_SHM",
            cache_servers=[{"host": "127.0.0.1", "port": 20000, "shard": 0}],
            distributed_servers=[{"host": "127.0.0.1", "port": 20000, "shard": 0}],
            distributed_num_shards=1,
        )
        fake_client = _FakeClientWithWriteRowLimit(max_rows_per_write=16_131)
        fake_client.ops.active_port = 20000
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        client.init_data(
            name="large_fused",
            shape=(20_000, 128),
            dtype=torch.float32,
        )

        self.assertGreater(len(fake_client.write_batch_sizes), 1)
        self.assertLessEqual(max(fake_client.write_batch_sizes), 16_131)
        self.assertEqual(sum(fake_client.write_batch_sizes), 20_000)
        ids = torch.tensor([0, 16_130, 16_131, 19_999], dtype=torch.int64)
        pulled = client.pull("large_fused", ids)
        self.assertTrue(torch.allclose(pulled, torch.zeros((4, 128))))

    def test_init_data_splits_large_writes_by_distributed_max_keys_per_request(self) -> None:
        runtime_dir = self._make_runtime_dir(
            cache_ps_type="GRPC",
            cache_servers=[{"host": "127.0.0.1", "port": 20000, "shard": 0}],
            distributed_servers=[{"host": "127.0.0.1", "port": 20000, "shard": 0}],
            distributed_num_shards=1,
            max_keys_per_request=500,
        )
        fake_client = _FakeClientWithWriteRowLimit(max_rows_per_write=500)
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        client.init_data(
            name="grpc_large_fused",
            shape=(1200, 128),
            dtype=torch.float32,
        )

        self.assertEqual(fake_client.write_batch_sizes, [500, 500, 200])
        ids = torch.tensor([0, 499, 500, 999, 1199], dtype=torch.int64)
        pulled = client.pull("grpc_large_fused", ids)
        self.assertTrue(torch.allclose(pulled, torch.zeros((5, 128))))

    def test_prefetch_wait_and_get_handles_colliding_shard_ids(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClientCollidingPrefetchId()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        keys = torch.tensor([5, 2, 7, 4], dtype=torch.int64)
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        client.emb_write(keys, values)

        handle = client.prefetch(keys)
        self.assertEqual(fake_client.prefetch_requests_by_port, {})
        result = client.wait_and_get(handle, 4)
        self.assertTrue(torch.allclose(result, values))
        self.assertEqual(
            sorted(fake_client.prefetch_requests_by_port.items()),
            [((20000, 1), [2, 4]), ((20001, 1), [5, 7])],
        )
        with self.assertRaises(RuntimeError):
            client.wait_and_get(handle, 4)

    def test_shared_local_shm_prefetch_uses_ready_lookup_result_once(self) -> None:
        runtime_dir = self._make_runtime_dir(
            cache_ps_type="LOCAL_SHM",
            cache_servers=[{"host": "127.0.0.1", "port": 20000, "shard": 0}],
            distributed_servers=[{"host": "127.0.0.1", "port": 20000, "shard": 0}],
            distributed_num_shards=1,
        )
        fake_client = _FakeClient()
        fake_client.ops.backend = "local_shm"
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table", shape=(16, 4), dtype=torch.float32)
        client.activate_shard(0)
        client.set_prefetch_table_name("table")

        ids = torch.tensor([3, 1, 3], dtype=torch.int64)
        handle = client.prefetch(ids)

        self.assertGreater(handle, 0)
        self.assertEqual(fake_client.prefetch_requests, {})
        self.assertEqual(fake_client.ops.lookup_calls, [(None, [3, 1, 3], 4)])
        result = client.wait_and_get(handle, 4)
        expected = torch.arange(12, dtype=torch.float32).view(3, 4)
        self.assertTrue(torch.allclose(result, expected))
        with self.assertRaises(RuntimeError):
            client.wait_and_get(handle, 4)

    def test_update_async_routes_updates_to_each_shard(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        client.init_data(name="default", shape=(8, 2), dtype=torch.float32)

        ids = torch.arange(8, dtype=torch.int64)
        grads = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        handle = client.update_async("default", ids, grads)
        client.wait(handle)

        port_to_ids: dict[int, list[int]] = {}
        for port, name, ids_list, _ in fake_client.updates:
            self.assertEqual(name, "default")
            port_to_ids.setdefault(port, []).extend(ids_list)

        self.assertEqual(sorted(port_to_ids[20000]), [0, 2, 4, 6])
        self.assertEqual(sorted(port_to_ids[20001]), [1, 3, 5, 7])

    def test_register_tensor_meta_allows_non_initializer_to_pull_and_update(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClient()
        initializer = ShardedRecstoreClient(fake_client, runtime_dir)
        follower = ShardedRecstoreClient(fake_client, runtime_dir)

        init_values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        initializer.init_data(
            name="default",
            shape=(4, 2),
            dtype=torch.float32,
            base_offset=50,
            init_func=lambda shape, dtype: init_values,
        )
        self.assertEqual(len(fake_client.table_inits), 2)

        follower.register_tensor_meta(
            name="default",
            shape=(4, 2),
            dtype=torch.float32,
            base_offset=50,
        )
        pulled = follower.pull("default", torch.arange(50, 54, dtype=torch.int64))
        self.assertTrue(torch.allclose(pulled, init_values))

        grads = torch.ones((4, 2), dtype=torch.float32)
        handle = follower.update_async(
            "default",
            torch.arange(50, 54, dtype=torch.int64),
            grads,
        )
        follower.wait(handle)

        self.assertEqual(len(fake_client.table_inits), 2)
        self.assertEqual(len(fake_client.updates), 2)

    def test_wait_clears_gpu_cache_after_fallback_update(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("default", shape=(8, 2), dtype=torch.float32)

        handle = client.update_async(
            "default",
            torch.arange(0, 4, dtype=torch.int64),
            torch.ones((4, 2), dtype=torch.float32),
        )
        client.wait(handle)

        self.assertEqual(len(fake_client.updates), 2)
        self.assertEqual(fake_client.ops.clear_gpu_cache_calls, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_wait_keeps_cuda_updates_for_precise_gpu_cache_invalidation(self) -> None:
        runtime_dir = self._make_runtime_dir(hash_method="simple_mod", distributed_num_shards=2)
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("default", shape=(8, 2), dtype=torch.float32)

        handle = client.update_async(
            "default",
            torch.arange(0, 4, dtype=torch.int64, device="cuda"),
            torch.ones((4, 2), dtype=torch.float32, device="cuda"),
        )
        client.wait(handle)

        self.assertEqual(len(fake_client.updates), 2)
        self.assertEqual(fake_client.ops.clear_gpu_cache_calls, 0)

    def test_current_and_set_ps_backend_forward_to_underlying_ops(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        self.assertEqual(client.current_ps_backend(), "brpc")

        client.set_ps_backend("brpc")

        self.assertEqual(client.current_ps_backend(), "brpc")
        self.assertEqual(fake_client.ops.backend_switch_calls, ["brpc"])

    def test_gpu_cache_api_is_exposed_and_forwards_to_underlying_client(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClientWithGpuCache()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        self.assertTrue(hasattr(ShardedRecstoreClient, "enable_gpu_cache"))
        self.assertTrue(client.enable_gpu_cache(1024, 4))

        self.assertEqual(fake_client.gpu_cache_calls, [(1024, 4)])
        self.assertEqual(
            client.get_last_gpu_cache_profile()["gpu_cache_backend_lookup_ms"],
            2.0,
        )
        client.set_gpu_cache_lookup_bypass_enabled(False)
        self.assertFalse(client.is_gpu_cache_lookup_bypass_enabled())

    def test_gpu_cache_enable_falls_back_to_underlying_ops(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClientWithOpsGpuCache()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        self.assertTrue(client.enable_gpu_cache(2048, 8))
        client.set_gpu_cache_lookup_bypass_enabled(False)

        self.assertEqual(fake_client.ops.gpu_cache_calls, [(2048, 8)])
        self.assertFalse(fake_client.ops.bypass_enabled)
        self.assertTrue(client.is_gpu_cache_lookup_bypassed())
        client.reset_gpu_cache_bypass_state()
        self.assertTrue(fake_client.ops.bypass_enabled)

    def test_apply_sgd_update_gpu_cache_forwards_to_underlying_ops(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClientWithOpsGpuCache()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 2), dtype=torch.float32)
        client._activate_shard(1)
        self.assertTrue(client.enable_gpu_cache(2048, 2))

        ids = torch.tensor([7, 3], dtype=torch.int32)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        ok = client.apply_sgd_update_gpu_cache(
            "table0",
            ids,
            grads,
            learning_rate=0.25,
        )

        self.assertTrue(ok)
        self.assertEqual(
            fake_client.ops.gpu_cache_sgd_update_calls,
            [([7, 3], [[1.0, 2.0], [3.0, 4.0]], 0.25)],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_invalidate_gpu_cache_forwards_to_underlying_ops(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClientWithOpsGpuCache()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 2), dtype=torch.float32)
        self.assertTrue(client.enable_gpu_cache(2048, 2))

        client.invalidate_gpu_cache("table0", torch.tensor([7, 3], dtype=torch.int64))

        self.assertEqual(fake_client.ops.gpu_cache_invalidate_calls, [[7, 3]])
        self.assertEqual(fake_client.ops.gpu_cache_invalidate_devices, ["cuda:0"])

    def test_apply_sgd_update_gpu_cache_prefers_ops_over_raw_client_wrapper(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClientWithClientAndOpsGpuCache()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 2), dtype=torch.float32)
        client._activate_shard(1)
        self.assertTrue(client.enable_gpu_cache(2048, 2))

        ok = client.apply_sgd_update_gpu_cache(
            "table0",
            torch.tensor([7], dtype=torch.int64),
            torch.ones((1, 2), dtype=torch.float32),
            learning_rate=0.5,
        )

        self.assertTrue(ok)
        self.assertEqual(
            fake_client.ops.gpu_cache_sgd_update_calls,
            [([7], [[1.0, 1.0]], 0.5)],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_apply_sgd_update_gpu_cache_moves_cpu_payload_to_cuda_ops(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClientWithOpsGpuCache()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 2), dtype=torch.float32)
        client._activate_shard(1)
        self.assertTrue(client.enable_gpu_cache(2048, 2))

        ok = client.apply_sgd_update_gpu_cache(
            "table0",
            torch.tensor([7], dtype=torch.int64),
            torch.ones((1, 2), dtype=torch.float32),
            learning_rate=0.5,
        )

        self.assertTrue(ok)
        self.assertEqual(fake_client.ops.gpu_cache_sgd_update_devices, [("cpu", "cuda:0")])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_apply_sgd_update_gpu_cache_does_not_require_active_network_shard(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClientWithOpsGpuCache()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 2), dtype=torch.float32)
        self.assertTrue(client.enable_gpu_cache(2048, 2))

        ok = client.apply_sgd_update_gpu_cache(
            "table0",
            torch.tensor([7], dtype=torch.int64),
            torch.ones((1, 2), dtype=torch.float32),
            learning_rate=0.5,
        )

        self.assertTrue(ok)
        self.assertEqual(
            fake_client.ops.gpu_cache_sgd_update_calls,
            [([7], [[1.0, 1.0]], 0.5)],
        )

    def test_gpu_cache_enable_requires_underlying_support(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        with self.assertRaisesRegex(RuntimeError, "enable_gpu_cache"):
            client.enable_gpu_cache(1024, 4)

    def test_local_lookup_flat_normalizes_ids_and_forwards_to_active_shard(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        fake_client.ops.backend = "local_shm"
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 4), dtype=torch.float32)
        client._activate_shard(1)

        ids = torch.tensor([7, 3], dtype=torch.int32)
        out = client.local_lookup_flat("table0", ids)

        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(fake_client.ops.lookup_calls, [(None, [7, 3], 4)])
        self.assertEqual(client._active_shard, 1)
        self.assertEqual(fake_client.ops.port_history, [])

    def test_local_lookup_flat_clears_gpu_cache_when_switching_tables(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        fake_client.ops.backend = "local_shm"
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 4), dtype=torch.float32)
        client.register_tensor_meta("table1", shape=(16, 4), dtype=torch.float32)
        client._activate_shard(1)

        client.local_lookup_flat("table0", torch.tensor([1], dtype=torch.int64))
        client.local_lookup_flat("table0", torch.tensor([2], dtype=torch.int64))
        client.local_lookup_flat("table1", torch.tensor([1], dtype=torch.int64))

        self.assertEqual(fake_client.ops.clear_gpu_cache_calls, 1)

    def test_local_update_flat_clears_gpu_cache_when_switching_tables(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        fake_client.ops.backend = "local_shm"
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 4), dtype=torch.float32)
        client.register_tensor_meta("table1", shape=(16, 4), dtype=torch.float32)
        client._activate_shard(1)

        client.local_lookup_flat("table0", torch.tensor([1], dtype=torch.int64))
        client.local_update_flat(
            "table1",
            torch.tensor([1], dtype=torch.int64),
            torch.ones((1, 4), dtype=torch.float32),
        )

        self.assertEqual(fake_client.ops.update_clear_gpu_cache_call_counts, [1])
        self.assertEqual(fake_client.ops.clear_gpu_cache_calls, 2)
        self.assertIsNone(client._gpu_cache_table_name)

    def test_activate_shard_skips_transport_reconfig_for_shared_local_shm_single_table(self) -> None:
        runtime_dir = self._make_runtime_dir(
            cache_ps_type="LOCAL_SHM",
            cache_servers=[
                {"host": "127.0.0.1", "port": 20000, "shard": 0},
            ],
        )
        cfg = json.loads((runtime_dir / "recstore_config.json").read_text(encoding="utf-8"))
        cfg["cache_ps"]["num_shards"] = 1
        (runtime_dir / "recstore_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        client._activate_shard(1)

        self.assertEqual(client._active_shard, 1)
        self.assertEqual(fake_client.ops.port_history, [])

    def test_activate_shard_keeps_transport_reconfig_for_non_shared_local_shm_runtime(self) -> None:
        runtime_dir = self._make_runtime_dir(cache_ps_type="LOCAL_SHM")
        cfg = json.loads((runtime_dir / "recstore_config.json").read_text(encoding="utf-8"))
        cfg["cache_ps"]["num_shards"] = 2
        cfg["cache_ps"]["servers"] = [
            {"host": "127.0.0.1", "port": 20000, "shard": 0},
            {"host": "127.0.0.1", "port": 20001, "shard": 1},
        ]
        (runtime_dir / "recstore_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        client._activate_shard(1)

        self.assertEqual(client._active_shard, 1)
        self.assertEqual(fake_client.ops.port_history, [20001])

    def test_reports_shared_local_shm_single_table_runtime_only_when_config_is_unambiguous(self) -> None:
        runtime_dir = self._make_runtime_dir(
            cache_ps_type="LOCAL_SHM",
            cache_servers=[
                {"host": "127.0.0.1", "port": 20000, "shard": 0},
            ],
        )
        cfg = json.loads((runtime_dir / "recstore_config.json").read_text(encoding="utf-8"))
        cfg["cache_ps"]["num_shards"] = 1
        (runtime_dir / "recstore_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        self.assertTrue(client.is_shared_local_shm_table())

    def test_reports_non_shared_runtime_when_local_shm_cache_exposes_multiple_shards(self) -> None:
        runtime_dir = self._make_runtime_dir(cache_ps_type="LOCAL_SHM")
        cfg = json.loads((runtime_dir / "recstore_config.json").read_text(encoding="utf-8"))
        cfg["cache_ps"]["num_shards"] = 2
        cfg["cache_ps"]["servers"] = [
            {"host": "127.0.0.1", "port": 20000, "shard": 0},
            {"host": "127.0.0.1", "port": 20001, "shard": 1},
        ]
        (runtime_dir / "recstore_config.json").write_text(json.dumps(cfg), encoding="utf-8")

        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)

        self.assertFalse(client.is_shared_local_shm_table())

    def test_local_update_flat_normalizes_inputs_and_forwards_to_active_shard(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        fake_client.ops.backend = "local_shm"
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 4), dtype=torch.float32)
        client._activate_shard(0)

        ids = torch.tensor([7, 3], dtype=torch.int32)
        grads = torch.ones((2, 4), dtype=torch.float64)
        client.local_update_flat("table0", ids, grads)

        self.assertEqual(
            fake_client.ops.update_calls,
            [(None, "table0", [7, 3], grads.to(dtype=torch.float32).tolist())],
        )
        self.assertEqual(client._active_shard, 0)
        self.assertEqual(fake_client.ops.port_history, [])

    def test_local_flat_ops_allow_hierkv_backend(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        fake_client.ops.backend = "hierkv"
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 4), dtype=torch.float32)
        client._activate_shard(0)

        ids = torch.tensor([7, 3], dtype=torch.int32)
        grads = torch.ones((2, 4), dtype=torch.float64)
        out = client.local_lookup_flat("table0", ids)
        client.local_update_flat("table0", ids, grads)

        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(fake_client.ops.lookup_calls, [(None, [7, 3], 4)])
        self.assertEqual(
            fake_client.ops.update_calls,
            [(None, "table0", [7, 3], grads.to(dtype=torch.float32).tolist())],
        )
        self.assertEqual(fake_client.ops.port_history, [])

    def test_warmup_local_lookup_flat_cuda_region_forwards_to_underlying_ops(self) -> None:
        runtime_dir = self._make_runtime_dir(
            cache_ps_type="LOCAL_SHM",
            cache_servers=[
                {"host": "127.0.0.1", "port": 20000, "shard": 0},
            ],
        )
        cfg = json.loads((runtime_dir / "recstore_config.json").read_text(encoding="utf-8"))
        cfg["cache_ps"]["num_shards"] = 1
        (runtime_dir / "recstore_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        fake_client = _FakeClient()
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client._activate_shard(0)

        ok = client.warmup_local_lookup_flat_cuda_region()

        self.assertTrue(ok)
        self.assertEqual(fake_client.ops.lookup_region_warmup_calls, 1)

    def test_local_flat_ops_fail_loudly_for_non_local_backend(self) -> None:
        runtime_dir = self._make_runtime_dir()
        fake_client = _FakeClient()
        fake_client.ops.backend = "brpc"
        client = ShardedRecstoreClient(fake_client, runtime_dir)
        client.register_tensor_meta("table0", shape=(16, 4), dtype=torch.float32)
        client._activate_shard(0)

        with self.assertRaisesRegex(RuntimeError, "local_shm or hierkv"):
            client.local_lookup_flat("table0", torch.tensor([1], dtype=torch.int64))
        with self.assertRaisesRegex(RuntimeError, "local_shm or hierkv"):
            client.local_update_flat(
                "table0",
                torch.tensor([1], dtype=torch.int64),
                torch.ones((1, 4), dtype=torch.float32),
            )

        self.assertEqual(fake_client.ops.lookup_calls, [])
        self.assertEqual(fake_client.ops.update_calls, [])


if __name__ == "__main__":
    unittest.main()
