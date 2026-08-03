import sys
import unittest
from pathlib import Path
from unittest import mock

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

    def current_ps_backend(self) -> str:
        return "local_shm"


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

    def test_fused_prefetch_reuses_slot_deduplication_metadata(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
        ]
        fake = _FakeOps()
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=_FakeKVClient(fake),
        )
        fused_ids = torch.tensor([1, 1, 3], dtype=torch.int64)
        result = ebc.issue_fused_id_prefetch(fused_ids, record_handle=False)
        handle, num_ids, issue_ts, unique_ids, inverse = result
        ebc.set_fused_prefetch_handle(
            handle,
            num_ids=num_ids,
            issue_ts=issue_ts,
            fused_ids_cpu=unique_ids,
            fused_inverse=inverse,
            full_batch=True,
        )

        with mock.patch.object(
            torch,
            "unique",
            side_effect=AssertionError("consume must reuse prefetch metadata"),
        ):
            embeddings, used = ebc._consume_fused_prefetch_embeddings(
                fused_ids,
                fused_ids,
                compute_device=torch.device("cpu"),
            )

        self.assertTrue(used)
        self.assertEqual(embeddings.shape, (3, 4))

    def test_record_pooled_grad_expands_bags_without_autograd(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=_FakeKVClient(_FakeOps()),
        )
        features = self._build_features()
        grad = torch.arange(16, dtype=torch.float32).view(2, 2, 4)

        ebc.record_pooled_grad(features, grad)

        trace = ebc._trace[0]
        expected_ids = torch.tensor(
            [1, 2, 3, 1 << 30, (1 << 30) + 4, (1 << 30) + 2]
        )
        expected_grads = torch.stack(
            [grad[0, 0], grad[0, 0], grad[1, 0], grad[0, 1], grad[1, 1], grad[1, 1]]
        )
        self.assertTrue(torch.equal(trace["ids"], expected_ids))
        self.assertTrue(torch.equal(trace["grads"], expected_grads))

    def test_record_pooled_grad_reuses_prepared_fused_ids(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=_FakeKVClient(_FakeOps()),
        )
        features = build_sparse_features(
            keys=["f1", "f2"],
            values=torch.tensor([1, 3, 0, 4], dtype=torch.int64),
            lengths=torch.ones(4, dtype=torch.int32),
        )
        grad = torch.arange(16, dtype=torch.float32).view(2, 2, 4)
        fused_ids = torch.tensor(
            [1, 3, 1 << 30, (1 << 30) + 4]
        )
        unique_ids, inverse = torch.unique(fused_ids, return_inverse=True)

        with mock.patch.object(
            torch,
            "unique",
            side_effect=AssertionError("gradient path must reuse prepared IDs"),
        ):
            ebc.record_pooled_grad(
                features,
                grad,
                prepared_ids=(unique_ids, inverse, fused_ids.numel()),
            )

        trace = ebc._trace[0]
        expected_grads = torch.zeros((unique_ids.numel(), 4))
        raw_grads = torch.stack(
            [grad[0, 0], grad[1, 0], grad[0, 1], grad[1, 1]]
        )
        expected_grads.index_add_(0, inverse, raw_grads)
        self.assertTrue(torch.equal(trace["ids"], unique_ids))
        self.assertTrue(torch.equal(trace["grads"], expected_grads))

    def test_prepare_prefetch_keeps_inverse_on_source_device(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=_FakeKVClient(_FakeOps()),
        )
        features = build_sparse_features(
            keys=["f1", "f2"],
            values=torch.tensor([1, 3, 1, 4], dtype=torch.int64),
            lengths=torch.ones(4, dtype=torch.int32),
        )
        unique_ids, inverse, raw_count = ebc.prepare_fused_prefetch(features)
        expected = torch.tensor([1, 3, (1 << 30) + 1, (1 << 30) + 4])
        expected_unique, expected_inverse = torch.unique(
            expected, return_inverse=True
        )
        self.assertEqual(raw_count, expected.numel())
        self.assertTrue(torch.equal(unique_ids, expected_unique))
        self.assertTrue(torch.equal(inverse, expected_inverse))
        self.assertEqual(inverse.device, features.device())
        if torch.cuda.is_available():
            features_cuda = build_sparse_features(
                keys=["f1", "f2"],
                values=torch.tensor(
                    [1, 3, 1, 4], dtype=torch.int64, device="cuda"
                ),
                lengths=torch.ones(4, dtype=torch.int32, device="cuda"),
            )
            gpu_unique, gpu_inverse, gpu_raw_count = ebc.prepare_fused_prefetch(
                features_cuda
            )
            self.assertEqual(gpu_raw_count, raw_count)
            self.assertTrue(torch.equal(gpu_unique, expected_unique))
            self.assertTrue(torch.equal(gpu_inverse.cpu(), expected_inverse))
            self.assertEqual(gpu_inverse.device.type, "cuda")

    def test_record_pooled_grad_falls_back_without_prepared_inverse(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=_FakeKVClient(_FakeOps()),
        )
        features = self._build_features()
        grad = torch.arange(16, dtype=torch.float32).view(2, 2, 4)
        ebc.record_pooled_grad(
            features,
            grad,
            prepared_ids=(torch.tensor([], dtype=torch.int64), None, 0),
        )
        self.assertEqual(len(ebc._trace), 1)
        trace = ebc._trace[0]
        self.assertEqual(trace["ids"].numel(), 6)
        self.assertEqual(trace["grads"].shape, (6, 4))

    def test_fused_prefetch_rejects_mismatched_slot(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
        ]
        fake = _FakeOps()
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=_FakeKVClient(fake),
        )
        prefetched_ids = torch.tensor([1, 1, 3], dtype=torch.int64)
        current_ids = torch.tensor([1, 3, 3], dtype=torch.int64)
        result = ebc.issue_fused_id_prefetch(prefetched_ids, record_handle=False)
        handle, num_ids, issue_ts, unique_ids, inverse = result
        ebc.set_fused_prefetch_handle(
            handle,
            num_ids=num_ids,
            issue_ts=issue_ts,
            fused_ids_cpu=unique_ids,
            fused_inverse=inverse,
            full_batch=True,
        )

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            ebc._consume_fused_prefetch_embeddings(
                current_ids,
                current_ids,
                compute_device=torch.device("cpu"),
            )

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

    def test_prepared_fused_prefetch_matches_sync(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=_FakeKVClient(fake),
        )
        for idx, cfg in enumerate(configs):
            base_offset = idx << 30
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            values = torch.arange(
                cfg["num_embeddings"] * cfg["embedding_dim"], dtype=torch.float32
            ).view(cfg["num_embeddings"], cfg["embedding_dim"])
            fake.emb_write(keys, values + idx * 1000)

        features = self._build_features()
        expected = ebc(features).values().detach().clone()
        prepared = ebc.prepare_fused_prefetch(features)
        ebc.issue_prepared_fused_prefetch(*prepared)
        actual = ebc(features).values().detach()

        self.assertTrue(torch.allclose(expected, actual))

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
        fake_client.is_shared_local_shm_table = lambda: True

        features = self._build_features()
        ebc.issue_fused_prefetch(features)
        ebc(features)

        self.assertEqual(fake_client.prefill_calls, [])
        self.assertIsNone(ebc._fused_prefetch_handle)
        self.assertEqual(ebc._fused_prefetch_slots, [])

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
        fake_client.is_shared_local_shm_table = lambda: True
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            keys = torch.arange(cfg["num_embeddings"], dtype=torch.int64) + base_offset
            vals = torch.ones((cfg["num_embeddings"], cfg["embedding_dim"]), dtype=torch.float32)
            fake.emb_write(keys, vals)

        features = self._build_features()
        ebc.issue_fused_prefetch(features)
        ebc(features)



if __name__ == "__main__":
    unittest.main()
