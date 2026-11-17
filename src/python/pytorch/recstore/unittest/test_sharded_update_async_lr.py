from __future__ import annotations

import unittest

import torch

from ..sharded_client import ShardServer, ShardedRecstoreClient


class _FakeOps:
    def __init__(self) -> None:
        self.shard_configs: list[tuple[str, int]] = []

    def current_ps_backend(self) -> str:
        return "brpc"

    def set_ps_config(self, host: str, port: int) -> None:
        self.shard_configs.append((host, port))


class _FakeClient:
    def __init__(self) -> None:
        self.ops = _FakeOps()
        self.updates: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []

    def emb_update_table(
        self,
        table_name: str,
        keys: torch.Tensor,
        grads: torch.Tensor,
        learning_rate: float = 0.0,
    ) -> None:
        self.updates.append((table_name, keys.clone(), grads.clone(), learning_rate))


class TestShardedUpdateAsyncLearningRate(unittest.TestCase):
    def _build_client(self) -> tuple[ShardedRecstoreClient, _FakeClient]:
        client = object.__new__(ShardedRecstoreClient)
        fake = _FakeClient()
        client._client = fake
        client._tensor_meta = {"table": {"shape": (16, 4), "dtype": torch.float32}}
        client._num_shards = 2
        client._hash_method = "simple_mod"
        client._native_distributed_backend = False
        client._cache_ps_type = "BRPC"
        client._cache_num_shards = 2
        client._cache_servers = []
        client._servers_by_shard = {
            0: ShardServer(0, "127.0.0.1", 15000),
            1: ShardServer(1, "127.0.0.1", 15001),
        }
        client._active_shard = None
        client._next_async_handle = 0
        client._pending_async_ops = {}
        client._clear_gpu_cache_after_cpu_update = False
        client._gpu_cache_table_name = None
        client._gpu_cache_clear_count = 0
        return client, fake

    def test_wait_groups_shards_and_forwards_learning_rate(self) -> None:
        client, fake = self._build_client()
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        grads = torch.ones((4, 4), dtype=torch.float32)

        handle = client.update_async("table", ids, grads, learning_rate=0.25)
        client.wait(handle)

        self.assertEqual(len(fake.updates), 2)
        self.assertEqual(
            {int(key) for _, keys, _, _ in fake.updates for key in keys.tolist()},
            {1, 2, 3, 4},
        )
        for _, _, _, learning_rate in fake.updates:
            self.assertEqual(learning_rate, 0.25)
        self.assertEqual(len(fake.ops.shard_configs), 2)


if __name__ == "__main__":
    unittest.main()
