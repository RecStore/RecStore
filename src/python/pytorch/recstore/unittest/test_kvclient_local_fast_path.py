import unittest

import torch

from ..KVClient import RecStoreClient


class _FakeOps:
    def __init__(self, backend: str = "local_shm"):
        self.backend = backend
        self.lookup_calls = []
        self.update_calls = []
        self.prefill_calls = []
        self.gpu_cache_sgd_update_calls = []
        self.emb_read_calls = []
        self.update_clear_gpu_cache_call_counts = []
        self.backend_switch_calls = []
        self.clear_gpu_cache_calls = 0

    def current_ps_backend(self) -> str:
        return self.backend

    def set_ps_backend(self, backend: str) -> None:
        self.backend = backend
        self.backend_switch_calls.append(backend)

    def local_lookup_flat(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.lookup_calls.append((keys.clone(), int(embedding_dim)))
        rows = keys.numel()
        return torch.arange(rows * int(embedding_dim), dtype=torch.float32).view(rows, int(embedding_dim))

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_calls.append((keys.clone(), int(embedding_dim)))
        rows = keys.numel()
        return torch.arange(rows * int(embedding_dim), dtype=torch.float32).view(rows, int(embedding_dim))

    def local_update_flat(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        self.update_clear_gpu_cache_call_counts.append(self.clear_gpu_cache_calls)
        self.update_calls.append((table_name, keys.clone(), grads.clone()))

    def prefill_gpu_cache(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.prefill_calls.append((keys.clone(), values.clone()))

    def apply_sgd_update_gpu_cache(
        self,
        keys: torch.Tensor,
        grads: torch.Tensor,
        learning_rate: float,
    ) -> bool:
        self.gpu_cache_sgd_update_calls.append(
            (keys.clone(), grads.clone(), float(learning_rate))
        )
        return True

    def clear_gpu_cache(self) -> None:
        self.clear_gpu_cache_calls += 1


class TestKVClientLocalFastPath(unittest.TestCase):
    def _build_client(self, backend: str = "local_shm") -> RecStoreClient:
        client = object.__new__(RecStoreClient)
        client.ops = _FakeOps(backend=backend)
        client._part_policy = {}
        client._tensor_meta = {"table_a": {"shape": (16, 4), "dtype": torch.float32}}
        client._full_data_shape = {"table_a": (16, 4)}
        client._data_name_list = {"table_a"}
        client._gdata_name_list = {"table_a"}
        client._role = "default"
        client._next_async_handle = 1
        client._pending_async_ops = {}
        client._gpu_cache_table_name = None
        client._initialized = True
        return client

    def test_local_lookup_flat_uses_explicit_local_op(self):
        client = self._build_client()

        keys = torch.tensor([7, 3], dtype=torch.int64)
        out = client.local_lookup_flat("table_a", keys)

        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(len(client.ops.lookup_calls), 1)
        called_keys, called_dim = client.ops.lookup_calls[0]
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertEqual(called_dim, 4)

    def test_local_update_flat_uses_explicit_local_op(self):
        client = self._build_client()

        keys = torch.tensor([7, 3], dtype=torch.int64)
        grads = torch.ones((2, 4), dtype=torch.float32)
        client.local_update_flat("table_a", keys, grads)

        self.assertEqual(len(client.ops.update_calls), 1)
        table_name, called_keys, called_grads = client.ops.update_calls[0]
        self.assertEqual(table_name, "table_a")
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertTrue(torch.equal(called_grads, grads))

    def test_prefill_gpu_cache_forwards_to_ops(self):
        client = self._build_client()

        keys = torch.tensor([7, 3], dtype=torch.int64)
        values = torch.ones((2, 4), dtype=torch.float32)
        client.prefill_gpu_cache("table_a", keys, values)

        self.assertEqual(len(client.ops.prefill_calls), 1)
        called_keys, called_values = client.ops.prefill_calls[0]
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertTrue(torch.equal(called_values, values))
        self.assertEqual(client._gpu_cache_table_name, "table_a")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU-cache-aware pull coverage")
    def test_pull_with_gpu_cache_keeps_cuda_ids(self):
        client = self._build_client(backend="BRPC")
        device = torch.device("cuda", 0)

        keys = torch.tensor([7, 3], dtype=torch.int64, device=device)
        client.pull_with_gpu_cache("table_a", keys)

        self.assertEqual(len(client.ops.emb_read_calls), 1)
        called_keys, called_dim = client.ops.emb_read_calls[0]
        self.assertEqual(called_keys.device.type, "cuda")
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertEqual(called_dim, 4)

    def test_apply_sgd_update_gpu_cache_forwards_to_ops(self):
        client = self._build_client()

        keys = torch.tensor([7, 3], dtype=torch.int64)
        grads = torch.ones((2, 4), dtype=torch.float32)
        self.assertTrue(
            client.apply_sgd_update_gpu_cache("table_a", keys, grads, learning_rate=0.25)
        )

        self.assertEqual(len(client.ops.gpu_cache_sgd_update_calls), 1)
        called_keys, called_grads, called_lr = client.ops.gpu_cache_sgd_update_calls[0]
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertTrue(torch.equal(called_grads, grads))
        self.assertEqual(called_lr, 0.25)
        self.assertEqual(client._gpu_cache_table_name, "table_a")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CPU-normalization regression coverage")
    def test_default_normalizers_still_move_cuda_tensors_to_cpu(self):
        client = self._build_client()
        device = torch.device("cuda", 0)

        ids = client._normalize_ids(torch.tensor([7, 3], dtype=torch.int64, device=device))
        grads = client._normalize_grads(
            torch.ones((2, 4), dtype=torch.float32, device=device)
        )

        self.assertEqual(ids.device.type, "cpu")
        self.assertEqual(grads.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU-resident local fast path coverage")
    def test_local_lookup_flat_keeps_cuda_ids_for_local_shm_backend(self):
        client = self._build_client()
        device = torch.device("cuda", 0)

        keys = torch.tensor([7, 3], dtype=torch.int64, device=device)
        client.local_lookup_flat("table_a", keys)

        self.assertEqual(len(client.ops.lookup_calls), 1)
        called_keys, called_dim = client.ops.lookup_calls[0]
        self.assertEqual(called_keys.device.type, "cuda")
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertEqual(called_dim, 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU-resident local fast path coverage")
    def test_local_update_flat_keeps_cuda_inputs_for_local_shm_backend(self):
        client = self._build_client()
        device = torch.device("cuda", 0)

        keys = torch.tensor([7, 3], dtype=torch.int64, device=device)
        grads = torch.ones((2, 4), dtype=torch.float32, device=device)
        client.local_update_flat("table_a", keys, grads)

        self.assertEqual(len(client.ops.update_calls), 1)
        table_name, called_keys, called_grads = client.ops.update_calls[0]
        self.assertEqual(table_name, "table_a")
        self.assertEqual(called_keys.device.type, "cuda")
        self.assertEqual(called_grads.device.type, "cuda")
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertTrue(torch.equal(called_grads, grads))

    def test_local_update_flat_clears_gpu_cache_when_switching_tables(self):
        client = self._build_client()
        client._tensor_meta["table_b"] = {"shape": (16, 4), "dtype": torch.float32}

        client.local_lookup_flat("table_a", torch.tensor([1], dtype=torch.int64))
        client.local_update_flat(
            "table_b",
            torch.tensor([1], dtype=torch.int64),
            torch.ones((1, 4), dtype=torch.float32),
        )

        self.assertEqual(client.ops.update_clear_gpu_cache_call_counts, [1])
        self.assertEqual(client.ops.clear_gpu_cache_calls, 2)
        self.assertIsNone(client._gpu_cache_table_name)

    def test_local_flat_ops_allow_hierkv_backend(self):
        client = self._build_client(backend="hierkv")

        keys = torch.tensor([7, 3], dtype=torch.int64)
        grads = torch.ones((2, 4), dtype=torch.float32)
        out = client.local_lookup_flat("table_a", keys)
        client.local_update_flat("table_a", keys, grads)

        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(len(client.ops.lookup_calls), 1)
        self.assertEqual(len(client.ops.update_calls), 1)

    def test_local_lookup_flat_fails_loudly_for_non_local_backend(self):
        client = self._build_client(backend="grpc")

        with self.assertRaisesRegex(RuntimeError, "local_shm or hierkv"):
            client.local_lookup_flat("table_a", torch.tensor([1], dtype=torch.int64))

        self.assertEqual(client.ops.lookup_calls, [])

    def test_local_update_flat_fails_loudly_for_non_local_backend(self):
        client = self._build_client(backend="brpc")

        with self.assertRaisesRegex(RuntimeError, "local_shm or hierkv"):
            client.local_update_flat(
                "table_a",
                torch.tensor([1], dtype=torch.int64),
                torch.ones((1, 4), dtype=torch.float32),
            )

        self.assertEqual(client.ops.update_calls, [])

    def test_local_fast_path_rejects_non_cpu_non_cuda_devices(self):
        client = self._build_client()

        meta_ids = torch.empty((1,), dtype=torch.int64, device="meta")
        meta_grads = torch.empty((1, 4), dtype=torch.float32, device="meta")

        with self.assertRaisesRegex(RuntimeError, "cpu or cuda"):
            client.local_lookup_flat("table_a", meta_ids)
        with self.assertRaisesRegex(RuntimeError, "cpu or cuda"):
            client.local_update_flat("table_a", meta_ids, meta_grads)

        self.assertEqual(client.ops.lookup_calls, [])
        self.assertEqual(client.ops.update_calls, [])

    def test_set_ps_backend_switches_backend_explicitly(self):
        client = self._build_client(backend="grpc")

        client.set_ps_backend("local_shm")

        self.assertEqual(client.ops.current_ps_backend(), "local_shm")
        self.assertEqual(client.ops.backend_switch_calls, ["local_shm"])

    def test_gpu_cache_control_ops_forward_to_library(self):
        client = self._build_client()
        client.ops.gpu_cache_calls = []
        client.ops.enable_gpu_cache = lambda capacity, embedding_dim: client.ops.gpu_cache_calls.append(
            ("enable", int(capacity), int(embedding_dim))
        ) or True
        client.ops.disable_gpu_cache = lambda: client.ops.gpu_cache_calls.append(("disable",))
        client.ops.clear_gpu_cache = lambda: client.ops.gpu_cache_calls.append(("clear",))
        client.ops.get_last_gpu_cache_profile = lambda: [1.0, 2.0, 3.0, 4.0, 5.0]
        client.ops.set_gpu_cache_lookup_bypass_enabled = (
            lambda enabled: client.ops.gpu_cache_calls.append(("bypass", bool(enabled)))
        )
        client.ops.is_gpu_cache_lookup_bypass_enabled = lambda: False
        client.ops.is_gpu_cache_lookup_bypassed = lambda: True
        client.ops.reset_gpu_cache_bypass_state = (
            lambda: client.ops.gpu_cache_calls.append(("reset_bypass",))
        )

        self.assertTrue(client.enable_gpu_cache(capacity=1024, embedding_dim=4))
        client.set_gpu_cache_lookup_bypass_enabled(False)
        self.assertFalse(client.is_gpu_cache_lookup_bypass_enabled())
        self.assertTrue(client.is_gpu_cache_lookup_bypassed())
        client.reset_gpu_cache_bypass_state()
        client.clear_gpu_cache()
        client.disable_gpu_cache()
        profile = client.get_last_gpu_cache_profile()

        self.assertEqual(
            client.ops.gpu_cache_calls,
            [
                ("enable", 1024, 4),
                ("bypass", False),
                ("reset_bypass",),
                ("clear",),
                ("disable",),
            ],
        )
        self.assertEqual(
            profile,
            {
                "gpu_cache_query_ms": 1.0,
                "gpu_cache_backend_lookup_ms": 2.0,
                "gpu_cache_fill_ms": 3.0,
                "gpu_cache_update_ms": 4.0,
                "gpu_cache_hit_count": 5.0,
                "gpu_cache_invalidate_ms": 0.0,
                "gpu_cache_request_count": 0.0,
                "gpu_cache_miss_count": 0.0,
            },
        )

    def test_gpu_cache_clears_when_local_lookup_switches_tables(self):
        client = self._build_client()
        client._tensor_meta["table_b"] = {"shape": (16, 4), "dtype": torch.float32}
        client.ops.clear_gpu_cache_calls = 0
        client.ops.clear_gpu_cache = lambda: setattr(
            client.ops,
            "clear_gpu_cache_calls",
            client.ops.clear_gpu_cache_calls + 1,
        )

        client.local_lookup_flat("table_a", torch.tensor([1], dtype=torch.int64))
        client.local_lookup_flat("table_a", torch.tensor([2], dtype=torch.int64))
        client.local_lookup_flat("table_b", torch.tensor([1], dtype=torch.int64))

        self.assertEqual(client.ops.clear_gpu_cache_calls, 1)
        self.assertEqual(client._gpu_cache_table_name, "table_b")

    def test_gpu_cache_control_requires_ops_support(self):
        client = self._build_client()
        client.ops.clear_gpu_cache = None
        client.ops.prefill_gpu_cache = None
        client.ops.set_gpu_cache_lookup_bypass_enabled = None
        client.ops.is_gpu_cache_lookup_bypass_enabled = None
        client.ops.is_gpu_cache_lookup_bypassed = None
        client.ops.reset_gpu_cache_bypass_state = None

        with self.assertRaisesRegex(RuntimeError, "enable_gpu_cache"):
            client.enable_gpu_cache(capacity=1024, embedding_dim=4)
        with self.assertRaisesRegex(RuntimeError, "disable_gpu_cache"):
            client.disable_gpu_cache()
        with self.assertRaisesRegex(RuntimeError, "clear_gpu_cache"):
            client.clear_gpu_cache()
        with self.assertRaisesRegex(RuntimeError, "prefill_gpu_cache"):
            client.prefill_gpu_cache(
                "table_a",
                torch.tensor([1], dtype=torch.int64),
                torch.ones((1, 4), dtype=torch.float32),
            )
        with self.assertRaisesRegex(RuntimeError, "set_gpu_cache_lookup_bypass_enabled"):
            client.set_gpu_cache_lookup_bypass_enabled(False)
        with self.assertRaisesRegex(RuntimeError, "is_gpu_cache_lookup_bypass_enabled"):
            client.is_gpu_cache_lookup_bypass_enabled()
        with self.assertRaisesRegex(RuntimeError, "is_gpu_cache_lookup_bypassed"):
            client.is_gpu_cache_lookup_bypassed()
        with self.assertRaisesRegex(RuntimeError, "reset_gpu_cache_bypass_state"):
            client.reset_gpu_cache_bypass_state()

        self.assertEqual(client.get_last_gpu_cache_profile(), {})

    def test_enable_gpu_cache_rejects_invalid_parameters_before_ops_call(self):
        client = self._build_client()
        client.ops.enable_gpu_cache_calls = []
        client.ops.enable_gpu_cache = lambda capacity, embedding_dim: client.ops.enable_gpu_cache_calls.append(
            (capacity, embedding_dim)
        ) or True

        invalid_args = (
            {"capacity": 0, "embedding_dim": 4, "message": "capacity"},
            {"capacity": 1024, "embedding_dim": 0, "message": "embedding_dim"},
            {"capacity": 1.5, "embedding_dim": 4, "message": "capacity"},
            {"capacity": 1024, "embedding_dim": "4", "message": "embedding_dim"},
        )
        for kwargs in invalid_args:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, kwargs["message"]):
                    client.enable_gpu_cache(
                        capacity=kwargs["capacity"],
                        embedding_dim=kwargs["embedding_dim"],
                    )

        self.assertEqual(client.ops.enable_gpu_cache_calls, [])


if __name__ == "__main__":
    unittest.main()
