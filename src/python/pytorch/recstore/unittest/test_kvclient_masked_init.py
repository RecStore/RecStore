import unittest

import torch

from ..KVClient import RecStoreClient


class _FakeOps:
    def __init__(self, with_found: bool = True, backend: str = "local_shm"):
        self.writes = []
        self.reads = []
        self.found_reads = []
        self.backend = backend
        if not with_found:
            return
        self.emb_read_with_found = self._emb_read_with_found

    def current_ps_backend(self) -> str:
        return self.backend

    def init_embedding_table(self, name, num_embeddings, embedding_dim, table_id):
        return 0

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.writes.append((keys.clone(), values.clone()))

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.reads.append(keys.clone())
        rows = keys.numel()
        return torch.zeros((rows, int(embedding_dim)), dtype=torch.float32)

    def _emb_read_with_found(self, keys: torch.Tensor, embedding_dim: int):
        self.found_reads.append(keys.clone())
        rows = keys.numel()
        values = torch.zeros((rows, int(embedding_dim)), dtype=torch.float32)
        found = torch.arange(rows) % 2 == 0
        return values, found


class TestKVClientMaskedInit(unittest.TestCase):
    def _build_client(
        self, with_found: bool = True, backend: str = "local_shm"
    ) -> RecStoreClient:
        client = object.__new__(RecStoreClient)
        client.ops = _FakeOps(with_found=with_found, backend=backend)
        client._part_policy = {}
        client._tensor_meta = {}
        client._full_data_shape = {}
        client._data_name_list = set()
        client._gdata_name_list = set()
        client._role = "default"
        client._next_table_id = 0
        client._next_async_handle = 1
        client._pending_async_ops = {}
        client._gpu_cache_table_name = None
        client._gpu_cache_enabled = False
        client._gpu_cache_clear_count = 0
        client._clear_gpu_cache_after_cpu_update = True
        client._prefetch_table_name = None
        client._perf_stats = {}
        client._initialized = True
        return client

    def test_init_data_mask_writes_only_selected_rows(self):
        client = self._build_client()
        client.init_data("table_a", (4, 2), torch.float32)

        init_keys, init_values = client.ops.writes[-1]
        self.assertEqual(init_keys.tolist(), [0, 1, 2, 3])
        self.assertEqual(init_values.tolist(), [[0.0, 0.0]] * 4)

    def test_init_data_mask_filters_keys_and_values(self):
        client = self._build_client()
        mask = torch.tensor([True, False, True, False])
        client.init_data("table_a", (4, 2), torch.float32, mask=mask)

        keys, values = client.ops.writes[-1]
        self.assertEqual(keys.tolist(), [0, 2])
        self.assertEqual(values.shape, (2, 2))

    def test_init_data_mask_respects_base_offset(self):
        client = self._build_client()
        mask = torch.tensor([False, True, False, True])
        client.init_data(
            "table_a", (4, 2), torch.float32, base_offset=10, mask=mask
        )

        keys, _ = client.ops.writes[-1]
        self.assertEqual(keys.tolist(), [11, 13])

    def test_init_data_mask_rejects_wrong_length(self):
        client = self._build_client()

        with self.assertRaisesRegex(ValueError, "mask"):
            client.init_data(
                "table_a",
                (4, 2),
                torch.float32,
                mask=torch.tensor([True, False]),
            )

    def test_init_data_mask_skipped_when_values_are_not_initialized(self):
        client = self._build_client()
        client.init_data(
            "table_a",
            (4, 2),
            torch.float32,
            initialize_values=False,
            mask=torch.tensor([True, False, False, False]),
        )

        self.assertEqual(client.ops.writes, [])

    def test_pull_with_found_uses_backend_mask(self):
        client = self._build_client()
        client.init_data("table_a", (4, 2), torch.float32)

        ids = torch.tensor([0, 1, 2], dtype=torch.int64)
        values, found = client.pull_with_found("table_a", ids)

        self.assertEqual(values.shape, (3, 2))
        self.assertIsNotNone(found)
        self.assertEqual(found.tolist(), [True, False, True])
        self.assertEqual(len(client.ops.found_reads), 1)
        self.assertEqual(client.ops.reads, [])

    def test_pull_with_found_falls_back_when_backend_lacks_it(self):
        client = self._build_client(with_found=False, backend="rdma")
        client.init_data("table_a", (4, 2), torch.float32)

        ids = torch.tensor([0, 1], dtype=torch.int64)
        values, found = client.pull_with_found("table_a", ids)

        self.assertEqual(values.shape, (2, 2))
        self.assertIsNone(found)
        self.assertEqual(len(client.ops.reads), 1)


if __name__ == "__main__":
    unittest.main()
