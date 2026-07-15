import unittest

import torch

from .. import KVClient as kv_client_module
from .. import optimizer as optimizer_module
from ..optimizer import SparseSGD
from ..single_node_exchange import SparseGradPayload


class _FakeLegacyKVClient:
    def __init__(self):
        self.update_async_calls = []
        self.wait_calls = []
        self.local_update_flat_calls = []
        self.set_ps_backend_calls = []
        self.activate_shard_calls = []
        self._next_handle = 1

    def update_async(self, name, ids, grads):
        handle = self._next_handle
        self._next_handle += 1
        self.update_async_calls.append((name, ids.clone(), grads.clone(), handle))
        return handle

    def wait(self, handle):
        self.wait_calls.append(int(handle))

    def local_update_flat(self, name, ids, grads):
        self.local_update_flat_calls.append((name, ids.clone(), grads.clone()))

    def set_ps_backend(self, backend):
        self.set_ps_backend_calls.append(str(backend))

    def activate_shard(self, shard):
        self.activate_shard_calls.append(int(shard))

    def is_shared_local_shm_table(self):
        return False


class _FakeModule:
    def __init__(self, trace, kv_client):
        self._config_names = ["table0"]
        self._trace = list(trace)
        self.kv_client = kv_client
        self.reset_trace_calls = 0

    def reset_trace(self):
        self.reset_trace_calls += 1
        self._trace = []


class _FakeFastPathModule(_FakeModule):
    def __init__(self, trace, kv_client, *, backend: str = "local_shm"):
        super().__init__(trace, kv_client)
        self.enable_single_node_distributed_fast_path = True
        self.single_node_distributed_mode = "single_node"
        self.single_node_owner_policy = "hash_mod_world_size"
        self.single_node_ps_backend = str(backend)


class _FakeSharedLocalShmDirectModule(_FakeFastPathModule):
    def __init__(self, trace, kv_client):
        super().__init__(trace, kv_client)
        self._enable_fusion = True
        self._master_config = type("MasterConfig", (), {"name": "table0"})()


class _FakeDist:
    def __init__(self, *, rank, world_size, initialized=True):
        self._rank = rank
        self._world_size = world_size
        self._initialized = initialized

    def is_initialized(self):
        return self._initialized

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


class TestSparseOptimizerSingleNodeDistributed(unittest.TestCase):
    def setUp(self):
        self._original_dist = optimizer_module.torch.distributed
        self._original_exchange_sparse_grads = getattr(
            optimizer_module,
            "exchange_sparse_grads",
            None,
        )
        self._original_get_kv_client = kv_client_module.get_kv_client

    def tearDown(self):
        optimizer_module.torch.distributed = self._original_dist
        if self._original_exchange_sparse_grads is None:
            delattr(optimizer_module, "exchange_sparse_grads")
        else:
            optimizer_module.exchange_sparse_grads = self._original_exchange_sparse_grads
        kv_client_module.get_kv_client = self._original_get_kv_client

    def test_init_uses_injected_module_kv_client_without_loading_global_client(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeModule(trace=[], kv_client=kv_client)

        def fail_get_kv_client():
            raise AssertionError("SparseSGD should use the module kv_client")

        kv_client_module.get_kv_client = fail_get_kv_client

        optimizer = SparseSGD([mod], lr=0.1)

        self.assertIs(optimizer.kv_client, kv_client)

    def test_fast_path_disabled_keeps_legacy_async_update_and_flush_wait(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeModule(
            trace=[
                (
                    "table0",
                    torch.tensor([5, 5, 8], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [2.0, 2.0],
                            [4.0, 4.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)

        optimizer.step()

        self.assertEqual(len(kv_client.update_async_calls), 1)
        self.assertEqual(kv_client.wait_calls, [])
        self.assertEqual(kv_client.local_update_flat_calls, [])
        _, ids, grads, handle = kv_client.update_async_calls[0]
        self.assertTrue(torch.equal(ids, torch.tensor([5, 8], dtype=torch.int64)))
        self.assertTrue(
            torch.allclose(
                grads,
                torch.tensor(
                    [
                        [3.0, 3.0],
                        [4.0, 4.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(handle, 1)
        self.assertEqual(mod.reset_trace_calls, 1)

        optimizer.flush()

        self.assertEqual(kv_client.wait_calls, [1])

    def test_fast_path_step_uses_owner_local_update_and_flush_is_noop(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([4, 5], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [10.0, 10.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        exchange_calls = []

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            exchange_calls.append((payload.clone(), world_size, backend))
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([4], dtype=torch.int64),
                    grads=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                ),
                SparseGradPayload(
                    rank=1,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([1], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([4], dtype=torch.int64),
                    grads=torch.tensor(
                        [
                            [2.5, 2.5],
                        ],
                        dtype=torch.float32,
                    ),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.step()

        self.assertEqual(len(exchange_calls), 1)
        self.assertEqual(len(kv_client.update_async_calls), 0)
        self.assertEqual(len(kv_client.local_update_flat_calls), 1)
        self.assertEqual(kv_client.wait_calls, [])
        self.assertEqual(kv_client.set_ps_backend_calls, ["local_shm"])
        self.assertEqual(kv_client.activate_shard_calls, [0])
        table_name, ids, grads = kv_client.local_update_flat_calls[0]
        self.assertEqual(table_name, "table0")
        self.assertTrue(torch.equal(ids, torch.tensor([4], dtype=torch.int64)))
        self.assertTrue(
            torch.allclose(
                grads,
                torch.tensor([[3.5, 3.5]], dtype=torch.float32),
            )
        )
        self.assertEqual(mod.reset_trace_calls, 1)

        optimizer.flush()

        self.assertEqual(kv_client.wait_calls, [])
        self.assertEqual(len(kv_client.local_update_flat_calls), 1)

    def test_shared_local_shm_single_table_step_bypasses_exchange_and_updates_locally(self):
        kv_client = _FakeLegacyKVClient()
        kv_client.is_shared_local_shm_table = lambda: True
        mod = _FakeSharedLocalShmDirectModule(
            trace=[
                (
                    "table0",
                    torch.tensor([4, 5, 4], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [10.0, 10.0],
                            [2.5, 2.5],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        def fail_exchange_sparse_grads(*args, **kwargs):
            raise AssertionError("shared local_shm single-table path should bypass sparse grad exchange")

        optimizer_module.exchange_sparse_grads = fail_exchange_sparse_grads

        optimizer.step()

        self.assertEqual(len(kv_client.update_async_calls), 0)
        self.assertEqual(len(kv_client.local_update_flat_calls), 1)
        self.assertEqual(kv_client.wait_calls, [])
        self.assertEqual(kv_client.set_ps_backend_calls, ["local_shm"])
        self.assertEqual(kv_client.activate_shard_calls, [0])
        table_name, ids, grads = kv_client.local_update_flat_calls[0]
        self.assertEqual(table_name, "table0")
        self.assertTrue(torch.equal(ids, torch.tensor([4, 5], dtype=torch.int64)))
        self.assertTrue(
            torch.allclose(
                grads,
                torch.tensor(
                    [
                        [3.5, 3.5],
                        [10.0, 10.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        payloads = optimizer.last_update_payloads()
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertIs(payload["module"], mod)
        self.assertEqual(payload["name"], "table0")
        self.assertEqual(payload["lr"], 0.1)
        self.assertTrue(torch.equal(payload["ids"], ids))
        self.assertTrue(torch.equal(payload["grads"], grads))
        profile = getattr(mod, "_single_node_fast_path_profile", None)
        self.assertIsInstance(profile, dict)
        self.assertEqual(profile.get("exchange_ms"), 0.0)
        self.assertEqual(profile.get("owner_aggregate_ms"), 0.0)
        self.assertIn("trace_collect_ms", profile)
        self.assertIn("trace_aggregate_ms", profile)
        self.assertGreaterEqual(profile.get("local_update_ms", -1.0), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU-resident sparse fast path coverage")
    def test_fast_path_step_keeps_owner_local_update_inputs_on_cuda(self):
        kv_client = _FakeLegacyKVClient()
        device = torch.device("cuda", 0)
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([4, 5], dtype=torch.int64, device=device),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [10.0, 10.0],
                        ],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                    source_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                    row_positions=torch.tensor([0], dtype=torch.int64, device=device),
                    fused_ids=torch.tensor([4], dtype=torch.int64, device=device),
                    grads=torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=device),
                ),
                SparseGradPayload(
                    rank=1,
                    destination_ranks=torch.tensor([0], dtype=torch.int64, device=device),
                    source_ranks=torch.tensor([1], dtype=torch.int64, device=device),
                    row_positions=torch.tensor([0], dtype=torch.int64, device=device),
                    fused_ids=torch.tensor([4], dtype=torch.int64, device=device),
                    grads=torch.tensor([[2.5, 2.5]], dtype=torch.float32, device=device),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.step()

        self.assertEqual(len(kv_client.local_update_flat_calls), 1)
        _, ids, grads = kv_client.local_update_flat_calls[0]
        self.assertEqual(ids.device.type, "cuda")
        self.assertEqual(grads.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU-resident sparse fast path coverage")
    def test_fast_path_step_avoids_python_list_rebuild_for_owner_ids(self):
        kv_client = _FakeLegacyKVClient()
        device = torch.device("cuda", 0)
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([4, 5], dtype=torch.int64, device=device),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [10.0, 10.0],
                        ],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        gathered_payloads = [
            SparseGradPayload(
                rank=0,
                destination_ranks=torch.tensor([0, 0], dtype=torch.int64, device=device),
                source_ranks=torch.tensor([0, 1], dtype=torch.int64, device=device),
                row_positions=torch.tensor([0, 1], dtype=torch.int64, device=device),
                fused_ids=torch.tensor([4, 4], dtype=torch.int64, device=device),
                grads=torch.tensor([[1.0, 1.0], [2.5, 2.5]], dtype=torch.float32, device=device),
            )
        ]

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            return gathered_payloads

        original_tensor_ctor = optimizer_module.torch.tensor

        def guarded_tensor(data, *args, **kwargs):
            if isinstance(data, list) and data and all(isinstance(v, int) for v in data):
                raise AssertionError("owner ids should stay tensor-native in sparse fast path")
            return original_tensor_ctor(data, *args, **kwargs)

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads
        optimizer_module.torch.tensor = guarded_tensor
        try:
            optimizer.step()
        finally:
            optimizer_module.torch.tensor = original_tensor_ctor

        self.assertEqual(len(kv_client.local_update_flat_calls), 1)
        _, ids, grads = kv_client.local_update_flat_calls[0]
        self.assertTrue(torch.equal(ids, torch.tensor([4], dtype=torch.int64, device=device)))
        self.assertTrue(torch.allclose(grads, torch.tensor([[3.5, 3.5]], dtype=torch.float32, device=device)))

    def test_fast_path_step_records_profile_breakdown_on_module(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([4, 5], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [10.0, 10.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([4], dtype=torch.int64),
                    grads=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                ),
                SparseGradPayload(
                    rank=1,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([1], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([4], dtype=torch.int64),
                    grads=torch.tensor([[2.5, 2.5]], dtype=torch.float32),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.step()

        profile = getattr(mod, "_single_node_fast_path_profile", None)
        self.assertIsInstance(profile, dict)
        for key in (
            "exchange_ms",
            "owner_aggregate_ms",
            "local_update_ms",
        ):
            self.assertIn(key, profile)
            self.assertGreaterEqual(profile[key], 0.0)

    def test_hierkv_fast_path_uses_same_owner_local_update_flow(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([6, 7], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [9.0, 9.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
            backend="hierkv",
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            del payload, world_size, backend
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([6], dtype=torch.int64),
                    grads=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                ),
                SparseGradPayload(
                    rank=1,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([1], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([6], dtype=torch.int64),
                    grads=torch.tensor([[4.0, 4.0]], dtype=torch.float32),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.step()

        self.assertEqual(len(kv_client.update_async_calls), 0)
        self.assertEqual(len(kv_client.local_update_flat_calls), 1)
        table_name, ids, grads = kv_client.local_update_flat_calls[0]
        self.assertEqual(table_name, "table0")
        self.assertTrue(torch.equal(ids, torch.tensor([6], dtype=torch.int64)))
        self.assertTrue(torch.allclose(grads, torch.tensor([[5.0, 5.0]], dtype=torch.float32)))

    def test_perf_stats_capture_fast_path_exchange_and_local_update(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([6, 7], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [9.0, 9.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
            backend="hierkv",
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            del payload, world_size, backend
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([6], dtype=torch.int64),
                    grads=torch.tensor([[5.0, 5.0]], dtype=torch.float32),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.reset_perf_stats()
        optimizer.step()
        stats = optimizer.consume_perf_stats()

        self.assertGreaterEqual(stats["update_trace_merge_ms"], 0.0)
        self.assertGreaterEqual(stats["update_owner_exchange_ms"], 0.0)
        self.assertGreaterEqual(stats["update_local_apply_ms"], 0.0)
        self.assertEqual(stats["update_async_enqueue_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
