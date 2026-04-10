from __future__ import annotations

import ctypes
import importlib
import json
import struct
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from model_zoo.rs_demo.runtime.recstore_distributed import (
    ShardedRecstoreClient,
)


class _FakeOps:
    def __init__(self) -> None:
        self.active_port: int | None = None
        self.port_history: list[int] = []

    def set_ps_config(self, host: str, port: int) -> None:
        self.active_port = int(port)
        self.port_history.append(int(port))


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
    def emb_prefetch(self, keys: torch.Tensor) -> int:
        assert self.ops.active_port is not None
        # Simulate backend with per-connection/local handle namespace:
        # different shards may return the same id.
        self.prefetch_requests[1] = (self.ops.active_port, [int(v) for v in keys.tolist()])
        return 1


class TestShardedRecstoreClient(unittest.TestCase):
    def _make_runtime_dir(
        self,
        *,
        hash_method: str = "simple_mod",
        distributed_servers: list[dict] | None = None,
        cache_servers: list[dict] | None = None,
        distributed_num_shards: int | None = None,
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
        if include_distributed_num_shards:
            cfg["distributed_client"]["num_shards"] = distributed_num_shards
        if include_distributed_servers:
            cfg["distributed_client"]["servers"] = distributed_servers
        (runtime_dir / "recstore_config.json").write_text(
            json.dumps(cfg),
            encoding="utf-8",
        )
        return runtime_dir

    @staticmethod
    def _cityhash_shard_for_key(key: int, num_shards: int) -> int:
        lib = ctypes.CDLL(
            "/app/RecStore/third_party/cityhash/src/.libs/libcityhash.so.0.0.0"
        )
        city_hash64 = lib._Z10CityHash64PKcm
        city_hash64.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        city_hash64.restype = ctypes.c_uint64
        raw = struct.pack("<Q", int(key) & 0xFFFFFFFFFFFFFFFF)
        return int(city_hash64(raw, len(raw)) % num_shards)

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
        out = client.emb_wait_result(pid, 4)
        self.assertTrue(torch.allclose(out, values))

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
        out = client.emb_wait_result(opaque_handle, 4)
        self.assertTrue(torch.allclose(out, values))

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


if __name__ == "__main__":
    unittest.main()
