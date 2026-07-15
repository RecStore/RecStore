import sys
import unittest
from pathlib import Path

import torch

_PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from pytorch.torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection
from model_zoo.rs_demo.data.dlrm_source import build_sparse_features


class _FakeOps:
    def __init__(self):
        self._store = {}  # id(int)->tensor(1,D)
        self._next_handle = 1
        self._prefetch_buf = {}

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor):
        keys = keys.to(torch.int64).cpu().contiguous()
        values = values.to(torch.float32).cpu().contiguous()
        assert values.dim() == 2
        for i in range(keys.numel()):
            self._store[int(keys[i].item())] = values[i:i+1].clone()

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        keys = keys.to(torch.int64).cpu().contiguous()
        out = torch.zeros((keys.numel(), int(embedding_dim)), dtype=torch.float32)
        for i in range(keys.numel()):
            kid = int(keys[i].item())
            if kid in self._store:
                row = self._store[kid]
                if row.size(1) != embedding_dim:
                    # simple reshape/pad-truncate to requested dim
                    if row.size(1) > embedding_dim:
                        out[i] = row[0, :embedding_dim]
                    else:
                        out[i, :row.size(1)] = row[0]
                else:
                    out[i] = row[0]
            else:
                # keep zeros for missing
                pass
        return out

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        # For testing, precompute the result with default dim 4; the real wait will pass dim anyway.
        keys = keys.to(torch.int64).cpu().contiguous()
        handle = self._next_handle
        self._next_handle += 1
        self._prefetch_buf[handle] = keys.clone()
        return handle

    def emb_wait_result(self, handle: int, embedding_dim: int) -> torch.Tensor:
        keys = self._prefetch_buf.pop(int(handle))
        return self.emb_read(keys, embedding_dim)


class _FakeKVClient:
    def __init__(self, ops: _FakeOps) -> None:
        self.ops = ops
        self._tensor_meta = {}
        self.prefill_calls = []
        self.local_lookup_calls = []
        self.use_prefill_values_for_local_lookup = False
        self.allow_cpu_gpu_cache_prefill = True
        self.fail_wait = False
        self.truncate_wait_rows = False

    def init_data(self, name, shape, dtype, base_offset: int = 0):
        self._tensor_meta[name] = {
            "shape": tuple(shape),
            "dtype": dtype,
            "base_offset": int(base_offset),
        }

    def pull(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        return self.ops.emb_read(ids, embedding_dim)

    def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        self.local_lookup_calls.append((name, ids.clone()))
        if self.use_prefill_values_for_local_lookup and self.prefill_calls:
            prefill_name, prefill_ids, prefill_values = self.prefill_calls[-1]
            self.assert_name = prefill_name
            index = {int(v.item()): i for i, v in enumerate(prefill_ids.cpu())}
            rows = [prefill_values[index[int(v.item())]] for v in ids.cpu()]
            return torch.stack(rows, dim=0).to(ids.device)
        return self.ops.emb_read(ids, embedding_dim).to(ids.device)

    def prefetch(self, ids: torch.Tensor) -> int:
        return int(self.ops.emb_prefetch(ids))

    def wait_and_get(self, handle: int, embedding_dim: int, device=torch.device("cpu")) -> torch.Tensor:
        if self.fail_wait:
            raise RuntimeError("injected wait failure")
        out = self.ops.emb_wait_result(handle, embedding_dim)
        if self.truncate_wait_rows and out.size(0) > 0:
            out = out[:-1].contiguous()
        if device.type == "cuda":
            out = out.to(device)
        return out

    def prefill_gpu_cache(self, name: str, ids: torch.Tensor, values: torch.Tensor) -> None:
        self.prefill_calls.append((name, ids.detach().clone(), values.detach().clone()))

    def is_shared_local_shm_table(self) -> bool:
        return False


class _FakeKVClientWithoutPrefill(_FakeKVClient):
    prefill_gpu_cache = None


class TestFusedPrefetch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _build_features(self):
        # Batch size = 2
        # f1: lengths [2,1] -> values 3 ids
        # f2: lengths [1,2] -> values 3 ids
        keys = ["f1", "f2"]
        values_f1 = torch.tensor([1, 2, 3], dtype=torch.int64)
        lengths_f1 = torch.tensor([2, 1], dtype=torch.int32)
        values_f2 = torch.tensor([0, 4, 2], dtype=torch.int64)
        lengths_f2 = torch.tensor([1, 2], dtype=torch.int32)
        values = torch.cat([values_f1, values_f2], dim=0)
        lengths = torch.cat([lengths_f1, lengths_f2], dim=0)
        kjt = build_sparse_features(keys=keys, values=values, lengths=lengths)
        return kjt

    def test_fused_prefetch_matches_sync(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        # mirror init_data writes
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            n, d = cfg["num_embeddings"], cfg["embedding_dim"]
            keys = torch.arange(n, dtype=torch.int64) + base_offset
            vals = torch.zeros((n, d), dtype=torch.float32)
            fake.emb_write(keys, vals)

        # Build features after initialization; rely on zero initialization for both tables.
        features = self._build_features()

        # Sync path (no prefetch)
        out_sync = ebc(features).values().detach().clone()

        # Fused prefetch path
        ebc.issue_fused_prefetch(features)
        out_prefetch = ebc(features).values().detach().clone()

        self.assertTrue(torch.allclose(out_sync, out_prefetch), "Fused prefetch output must match sync output (zero-init case)")

        stats = ebc.report_prefetch_stats(reset=True)
        self.assertGreaterEqual(stats.get("batches_prefetched", 0), 1)

    def test_empty_fused_id_prefetch_does_not_call_backend(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )

        result = ebc.issue_fused_id_prefetch(
            torch.empty((0,), dtype=torch.int64),
            record_handle=False,
        )

        handle, num_ids, _, unique_ids, inverse = result
        self.assertEqual(handle, 0)
        self.assertEqual(num_ids, 0)
        self.assertEqual(unique_ids.numel(), 0)
        self.assertEqual(inverse.numel(), 0)
        self.assertEqual(fake._prefetch_buf, {})

    def test_partial_fused_id_prefetch_merges_with_local_lookup(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = (
                torch.arange(cfg["num_embeddings"] * cfg["embedding_dim"], dtype=torch.float32)
                .view(cfg["num_embeddings"], cfg["embedding_dim"])
                + idx * 1000
            )
            fake.emb_write(keys, vals)

        features = self._build_features()
        out_sync = ebc(features).values().detach().clone()
        ebc.consume_perf_stats(reset=True)

        partial_ids = torch.tensor([1, (1 << 30) + 4], dtype=torch.int64)
        ebc.issue_fused_id_prefetch(partial_ids)
        out_prefetch = ebc(features).values().detach().clone()

        self.assertTrue(torch.allclose(out_sync, out_prefetch))
        self.assertEqual(len(fake_client.local_lookup_calls), 1)
        looked_up_ids = set(fake_client.local_lookup_calls[0][1].tolist())
        self.assertNotIn(1, looked_up_ids)
        self.assertNotIn((1 << 30) + 4, looked_up_ids)
        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["lookup_fallback_pull_ms"], 0.0)

    def test_partial_fused_prefetch_records_merge_stats(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = (
                torch.arange(cfg["num_embeddings"] * cfg["embedding_dim"], dtype=torch.float32)
                .view(cfg["num_embeddings"], cfg["embedding_dim"])
                + idx * 1000
            )
            fake.emb_write(keys, vals)

        features = self._build_features()
        out_sync = ebc(features).values().detach().clone()
        ebc.consume_perf_stats(reset=True)

        partial_ids = torch.tensor([1, (1 << 30) + 4], dtype=torch.int64)
        ebc.issue_fused_id_prefetch(partial_ids)
        out_prefetch = ebc(features).values().detach().clone()

        self.assertTrue(torch.allclose(out_sync, out_prefetch))
        perf = ebc.consume_perf_stats(reset=True)
        self.assertGreater(perf["lookup_partial_merge_ms"], 0.0)
        self.assertEqual(perf["lookup_partial_prefetched_ids"], 2.0)
        self.assertEqual(perf["lookup_partial_missing_ids"], 4.0)

    def test_invalid_fused_prefetch_ids_are_refreshed_from_local_lookup(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = torch.zeros((cfg["num_embeddings"], cfg["embedding_dim"]), dtype=torch.float32)
            fake.emb_write(keys, vals)

        features = self._build_features()
        result = ebc.issue_fused_prefetch(features, record_handle=False)
        handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = result

        fake.emb_write(
            torch.tensor([1], dtype=torch.int64),
            torch.full((1, 4), 10.0, dtype=torch.float32),
        )
        ebc.set_fused_prefetch_handle(
            handle,
            num_ids=num_ids,
            issue_ts=issue_ts,
            fused_ids_cpu=fused_ids_cpu,
            fused_inverse=fused_inverse,
            invalid_fused_ids_cpu=torch.tensor([1], dtype=torch.int64),
        )
        out = ebc(features).values().detach()

        self.assertTrue(torch.allclose(out[0, :4], torch.full((4,), 10.0)))
        self.assertEqual(len(fake_client.local_lookup_calls), 1)
        self.assertEqual(fake_client.local_lookup_calls[0][1].tolist(), [1])

    def test_partial_fused_prefetch_falls_back_to_subset_pull_when_local_lookup_unavailable(self):
        class _NoLocalLookupClient(_FakeKVClient):
            def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
                self.local_lookup_calls.append((name, ids.clone()))
                raise RuntimeError("local_lookup_flat requires an active shard")

        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _NoLocalLookupClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = (
                torch.arange(cfg["num_embeddings"] * cfg["embedding_dim"], dtype=torch.float32)
                .view(cfg["num_embeddings"], cfg["embedding_dim"])
                + idx * 1000
            )
            fake.emb_write(keys, vals)

        features = self._build_features()
        out_sync = ebc(features).values().detach().clone()
        ebc.consume_perf_stats(reset=True)

        ebc.issue_fused_id_prefetch(torch.tensor([1, (1 << 30) + 4], dtype=torch.int64))
        out_prefetch = ebc(features).values().detach().clone()

        self.assertTrue(torch.allclose(out_sync, out_prefetch))
        self.assertEqual(len(fake_client.local_lookup_calls), 1)
        perf = ebc.consume_perf_stats(reset=True)
        self.assertGreater(perf["lookup_fallback_pull_ms"], 0.0)

    def test_partial_fused_prefetch_uses_gpu_cache_aware_pull_when_available(self):
        class _GpuCacheAwarePullClient(_FakeKVClient):
            def __init__(self, ops: _FakeOps) -> None:
                super().__init__(ops)
                self.gpu_cached_pull_calls = []

            def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
                self.local_lookup_calls.append((name, ids.clone()))
                raise RuntimeError("local_lookup_flat requires an active shard")

            def pull_with_gpu_cache(self, name: str, ids: torch.Tensor) -> torch.Tensor:
                self.gpu_cached_pull_calls.append((name, ids.clone()))
                embedding_dim = int(self._tensor_meta[name]["shape"][1])
                return self.ops.emb_read(ids, embedding_dim).to(ids.device)

        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _GpuCacheAwarePullClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = (
                torch.arange(cfg["num_embeddings"] * cfg["embedding_dim"], dtype=torch.float32)
                .view(cfg["num_embeddings"], cfg["embedding_dim"])
                + idx * 1000
            )
            fake.emb_write(keys, vals)

        features = self._build_features()
        out_sync = ebc(features).values().detach().clone()
        ebc.consume_perf_stats(reset=True)
        fake_client.gpu_cached_pull_calls.clear()

        ebc.issue_fused_id_prefetch(torch.tensor([1, (1 << 30) + 4], dtype=torch.int64))
        out_prefetch = ebc(features).values().detach().clone()

        self.assertTrue(torch.allclose(out_sync, out_prefetch))
        self.assertEqual(len(fake_client.local_lookup_calls), 1)
        self.assertEqual(len(fake_client.gpu_cached_pull_calls), 1)
        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["lookup_fallback_pull_ms"], 0.0)

    def test_fused_fallback_without_prefetch_uses_gpu_cache_aware_pull_when_available(self):
        class _GpuCacheAwarePullClient(_FakeKVClient):
            def __init__(self, ops: _FakeOps) -> None:
                super().__init__(ops)
                self.gpu_cached_pull_calls = []

            def pull_with_gpu_cache(self, name: str, ids: torch.Tensor) -> torch.Tensor:
                self.gpu_cached_pull_calls.append((name, ids.clone()))
                embedding_dim = int(self._tensor_meta[name]["shape"][1])
                return self.ops.emb_read(ids, embedding_dim).to(ids.device)

        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _GpuCacheAwarePullClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = (
                torch.arange(cfg["num_embeddings"] * cfg["embedding_dim"], dtype=torch.float32)
                .view(cfg["num_embeddings"], cfg["embedding_dim"])
                + idx * 1000
            )
            fake.emb_write(keys, vals)

        features = self._build_features()
        out = ebc(features).values().detach().clone()

        self.assertGreater(out.numel(), 0)
        self.assertEqual(len(fake_client.gpu_cached_pull_calls), 1)
        perf = ebc.consume_perf_stats(reset=True)
        self.assertGreater(perf["lookup_gpu_cache_pull_ms"], 0.0)
        self.assertEqual(perf["lookup_fallback_pull_ms"], 0.0)

    def test_fused_prefetch_prefills_gpu_cache_before_local_lookup(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        fake_client.use_prefill_values_for_local_lookup = True
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        fake_client.is_shared_local_shm_table = lambda: True
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            n, d = cfg["num_embeddings"], cfg["embedding_dim"]
            keys = torch.arange(n, dtype=torch.int64) + base_offset
            vals = torch.arange(n * d, dtype=torch.float32).view(n, d) + idx * 1000
            fake.emb_write(keys, vals)

        features = self._build_features()
        ebc.issue_fused_prefetch(features)
        out = ebc(features)

        self.assertEqual(len(fake_client.prefill_calls), 1)
        self.assertEqual(len(fake_client.local_lookup_calls), 1)
        name, prefill_ids, prefill_values = fake_client.prefill_calls[0]
        self.assertEqual(name, "t0")
        self.assertEqual(prefill_ids.dtype, torch.int64)
        self.assertEqual(prefill_values.dtype, torch.float32)
        self.assertEqual(prefill_ids.device.type, prefill_values.device.type)
        looked_up_ids = fake_client.local_lookup_calls[0][1]
        self.assertTrue(set(looked_up_ids.cpu().tolist()).issubset(set(prefill_ids.cpu().tolist())))
        self.assertEqual(out.values().shape, (2, 8))
        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["planned_gpu_cache_prefill_batches"], 1.0)
        self.assertGreater(perf["planned_gpu_cache_prefill_ids"], 0.0)
        self.assertEqual(perf["planned_gpu_cache_prefill_successes"], 1.0)
        self.assertEqual(perf["planned_gpu_cache_prefill_fallbacks"], 0.0)
        self.assertGreater(perf["lookup_total_ms"], 0.0)

    def test_fused_prefetch_prefill_disabled_on_cpu_without_test_override(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        fake_client.allow_cpu_gpu_cache_prefill = False
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = torch.zeros((cfg["num_embeddings"], cfg["embedding_dim"]), dtype=torch.float32)
            fake.emb_write(keys, vals)
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        fake_client.is_shared_local_shm_table = lambda: True

        features = self._build_features()
        ebc.issue_fused_prefetch(features)
        ebc(features)

        self.assertEqual(fake_client.prefill_calls, [])
        self.assertIsNone(ebc._fused_prefetch_handle)
        self.assertEqual(ebc._fused_prefetch_slots, [])
        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["planned_gpu_cache_prefill_no_cuda"], 1.0)
        self.assertEqual(perf["planned_gpu_cache_prefill_fallbacks"], 1.0)

    def test_fused_prefetch_prefill_falls_back_without_prefill_api(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClientWithoutPrefill(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        fake_client.is_shared_local_shm_table = lambda: True
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = torch.zeros((cfg["num_embeddings"], cfg["embedding_dim"]), dtype=torch.float32)
            fake.emb_write(keys, vals)

        features = self._build_features()
        ebc.issue_fused_prefetch(features)
        ebc(features)

        self.assertIsNone(ebc._fused_prefetch_handle)
        self.assertEqual(ebc._fused_prefetch_slots, [])
        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["planned_gpu_cache_prefill_no_api"], 1.0)
        self.assertEqual(perf["planned_gpu_cache_prefill_fallbacks"], 1.0)

    def test_fused_prefetch_prefill_falls_back_to_local_lookup_on_wait_failure(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        fake_client.fail_wait = True
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        fake_client.is_shared_local_shm_table = lambda: True
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = torch.arange(cfg["num_embeddings"] * cfg["embedding_dim"], dtype=torch.float32).view(
                cfg["num_embeddings"], cfg["embedding_dim"]
            )
            fake.emb_write(keys, vals)

        features = self._build_features()
        ebc.issue_fused_prefetch(features)
        ebc(features)

        self.assertEqual(len(fake_client.prefill_calls), 1)
        self.assertEqual(len(fake_client.local_lookup_calls), 2)
        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["planned_gpu_cache_prefill_fallbacks"], 1.0)
        self.assertEqual(perf["planned_gpu_cache_prefill_wait_failures"], 1.0)
        self.assertEqual(perf["planned_gpu_cache_prefill_successes"], 1.0)

    def test_fused_prefetch_prefill_records_size_mismatch_fallback(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        fake_client.truncate_wait_rows = True
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        fake_client.is_shared_local_shm_table = lambda: True
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = torch.ones((cfg["num_embeddings"], cfg["embedding_dim"]), dtype=torch.float32)
            fake.emb_write(keys, vals)

        features = self._build_features()
        ebc.issue_fused_prefetch(features)
        ebc(features)

        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["planned_gpu_cache_prefill_result_size_mismatches"], 1.0)
        self.assertEqual(perf["planned_gpu_cache_prefill_fallbacks"], 1.0)


if __name__ == "__main__":
    unittest.main()
