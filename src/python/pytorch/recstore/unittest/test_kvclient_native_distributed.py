import unittest

import torch

from ..KVClient import RecStoreClient


class _FakeDistributedOps:
    def __init__(self) -> None:
        self.read_calls = []
        self.write_calls = []
        self.update_calls = []
        self.init_calls = []

    def current_ps_backend(self) -> str:
        return "distributed_brpc"

    def init_embedding_table(
        self, table_name: str, num_embeddings: int, embedding_dim: int, table_id: int = 0
    ) -> int:
        self.init_calls.append((table_name, num_embeddings, embedding_dim))
        return int(table_id)

    def emb_write(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        self.write_calls.append((ids.clone(), values.clone()))

    def emb_read(self, ids: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.read_calls.append((ids.clone(), embedding_dim))
        return ids.to(torch.float32).unsqueeze(1).repeat(1, embedding_dim)

    def emb_update_table(
        self,
        table_name: str,
        ids: torch.Tensor,
        grads: torch.Tensor,
        learning_rate: float = 0.0,
    ) -> None:
        self.update_calls.append((table_name, ids.clone(), grads.clone()))


class TestKVClientNativeDistributed(unittest.TestCase):
    def setUp(self) -> None:
        self.client = object.__new__(RecStoreClient)
        self.client.ops = _FakeDistributedOps()
        self.client._part_policy = {}
        self.client._tensor_meta = {}
        self.client._full_data_shape = {}
        self.client._data_name_list = set()
        self.client._gdata_name_list = set()
        self.client._role = "default"
        self.client._next_async_handle = 1
        self.client._pending_async_ops = {}
        self.client._gpu_cache_table_name = None
        self.client._gpu_cache_clear_count = 0
        self.client._next_table_id = 0
        self.client._perf_stats = {}
        self.client._initialized = True

    def test_init_data_registers_metadata_and_writes_once(self) -> None:
        self.client.init_data(
            "table",
            (4, 2),
            torch.float32,
            base_offset=100,
        )

        self.assertEqual(self.client.ops.init_calls, [("table", 4, 2)])
        self.assertEqual(self.client._tensor_meta["table"]["base_offset"], 100)
        self.assertEqual(len(self.client.ops.write_calls), 1)
        ids, values = self.client.ops.write_calls[0]
        self.assertTrue(torch.equal(ids, torch.tensor([100, 101, 102, 103])))
        self.assertTrue(torch.equal(values, torch.zeros((4, 2))))

    def test_pull_into_owns_destination_and_normalizes_namespace(self) -> None:
        self.client.register_tensor_meta("table", (16, 2), torch.float32)
        self.client._tensor_meta["table"]["tag"] = 3
        ids = torch.tensor([4, 1, 4], dtype=torch.int64)
        destination = torch.empty((3, 2))
        for native in (False, True):
            calls = []
            if native:
                def read_into(keys, values):
                    calls.append((keys.clone(), values.data_ptr()))
                    values.copy_(torch.tensor([[4., 8.], [1., 2.], [4., 8.]]))
                self.client.ops.emb_read_into = read_into
            returned = self.client.pull_into("table", ids, destination)
            self.assertIs(returned, destination)
            expected_keys = ids | (3 << 56)
            if native:
                self.assertTrue(torch.equal(calls[0][0], expected_keys))
                self.assertEqual(calls[0][1], destination.data_ptr())
                torch.testing.assert_close(destination[:, 0], ids.float())
            else:
                self.assertTrue(torch.equal(self.client.ops.read_calls[-1][0], expected_keys))
                torch.testing.assert_close(destination[:, 0], expected_keys.float())

    def test_pull_into_rejects_invalid_destination_before_read(self) -> None:
        self.client.register_tensor_meta("table", (16, 2), torch.float32)
        ids = torch.tensor([1, 2], dtype=torch.int64)
        for values in (torch.empty((2, 3)), torch.empty((2, 2), dtype=torch.float64),
                       torch.empty((2, 2)).t(), torch.empty((2, 2), requires_grad=True)):
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaises(ValueError):
                    self.client.pull_into("table", ids, values)
        self.assertEqual(self.client.ops.read_calls, [])
        with self.assertRaises(RuntimeError):
            self.client.pull_into("missing", ids, torch.empty((2, 2)))

    def test_native_backend_requests_are_forwarded_without_python_sharding(self) -> None:
        self.client.register_tensor_meta("table", (16, 2), torch.float32)
        ids = torch.tensor([9, 2, 11, 4], dtype=torch.int64)
        grads = torch.ones((4, 2), dtype=torch.float32)

        values = self.client.pull("table", ids)
        self.client.update("table", ids, grads)

        self.assertEqual(self.client.current_ps_backend(), "distributed_brpc")
        self.assertEqual(len(self.client.ops.read_calls), 1)
        self.assertTrue(torch.equal(self.client.ops.read_calls[0][0], ids))
        self.assertTrue(torch.equal(values[:, 0], ids.to(torch.float32)))
        self.assertEqual(len(self.client.ops.update_calls), 1)
        self.assertTrue(torch.equal(self.client.ops.update_calls[0][1], ids))


if __name__ == "__main__":
    unittest.main()
