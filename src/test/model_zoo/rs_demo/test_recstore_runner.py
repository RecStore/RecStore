from __future__ import annotations

import contextlib
import io
import csv
import math
from contextlib import ExitStack
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from model_zoo.rs_demo import config
from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners import recstore_runner
from model_zoo.rs_demo.runners.recstore_runner import (
    RecStoreRunner,
    _build_train_dataloader_for_mode,
    _maybe_wrap_dense_module_for_dist,
)
from recstore.embedding_read_path import LookaheadPrefetcher


class _DummyDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(17, 1)

    def forward(self, dense_features: torch.Tensor, embedded_sparse: torch.Tensor) -> torch.Tensor:
        flat_sparse = embedded_sparse.reshape(embedded_sparse.shape[0], -1)
        features = torch.cat([dense_features, flat_sparse], dim=1).to(self.linear.weight.device)
        return self.linear(features)


class _FakeRecStoreClient:
    def __init__(self) -> None:
        self.emb_read_calls = 0
        self.emb_read_prefetch_calls = 0
        self.emb_prefetch_calls = 0
        self.emb_wait_result_calls = 0
        self.init_embedding_table_calls = 0
        self.emb_write_calls = 0
        self.set_ps_backend_calls: list[str] = []
        self.activate_shard_calls: list[int] = []
        self.enable_gpu_cache_calls: list[tuple[int, int]] = []
        self.enable_gpu_cache_result = True
        self.gpu_cache_enabled = False
        self.gpu_cache_lookup_bypass_enabled: bool | None = True
        self.gpu_cache_clear_count = 0
        self.clear_after_cpu_update_flags: list[bool] = []
        self.gpu_cache_sgd_update_calls = []
        self.local_shm_warmup_calls = 0
        self._shared_local_shm_table = False
        self._last_prefetch_keys = torch.empty((0,), dtype=torch.int64)
        self._current_ps_backend = "local_shm"

    def set_ps_backend(self, backend: str) -> None:
        backend = str(backend)
        self.set_ps_backend_calls.append(backend)
        self._current_ps_backend = backend

    def current_ps_backend(self) -> str:
        return self._current_ps_backend

    def activate_shard(self, shard: int) -> None:
        self.activate_shard_calls.append(int(shard))

    def enable_gpu_cache(self, capacity: int, embedding_dim: int) -> bool:
        self.enable_gpu_cache_calls.append((int(capacity), int(embedding_dim)))
        self.gpu_cache_enabled = bool(self.enable_gpu_cache_result)
        return self.enable_gpu_cache_result

    def is_gpu_cache_enabled(self) -> bool:
        return self.gpu_cache_enabled

    def set_gpu_cache_lookup_bypass_enabled(self, enabled: bool) -> None:
        self.gpu_cache_lookup_bypass_enabled = bool(enabled)

    def get_gpu_cache_clear_count(self) -> int:
        return int(self.gpu_cache_clear_count)

    def set_clear_gpu_cache_after_cpu_update(self, enabled: bool) -> None:
        self.clear_after_cpu_update_flags.append(bool(enabled))

    def apply_sgd_update_gpu_cache(self, name, ids, grads, *, learning_rate) -> bool:
        self.gpu_cache_sgd_update_calls.append(
            (
                str(name),
                ids.detach().to(dtype=torch.int64, device="cpu").clone(),
                grads.detach().to(dtype=torch.float32, device="cpu").clone(),
                float(learning_rate),
            )
        )
        return True

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        self.init_embedding_table_calls += 1
        return True

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.emb_write_calls += 1
        return None

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_calls += 1
        raise AssertionError("prefetch read mode should not call emb_read")

    def emb_read_prefetch(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_prefetch_calls += 1
        return torch.zeros((keys.numel(), embedding_dim), dtype=torch.float32)

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        self.emb_prefetch_calls += 1
        raise AssertionError("prefetch read mode should use stable emb_read_prefetch")

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        self.emb_wait_result_calls += 1
        raise AssertionError("prefetch read mode should use stable emb_read_prefetch")

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        return None

    def warmup_local_lookup_flat_cuda_region(self) -> bool:
        self.local_shm_warmup_calls += 1
        return True

    def is_shared_local_shm_table(self) -> bool:
        return self._shared_local_shm_table


class _FakeDirectReadRecStoreClient(_FakeRecStoreClient):
    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_calls += 1
        return torch.zeros((keys.numel(), embedding_dim), dtype=torch.float32)


class _FakeRecStoreEmbeddingBagCollection:
    last_instance = None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.kv_client = kwargs.get("kv_client")
        self._enable_fusion = bool(kwargs.get("enable_fusion", True))
        self.issue_fused_prefetch_calls = 0
        self.issue_fused_prefetch_record_flags: list[bool] = []
        self.issue_fused_prefetch_features: list[object] = []
        self.issue_fused_id_prefetch_ids: list[torch.Tensor] = []
        self.prepare_fused_prefetch_calls = 0
        self.set_fused_prefetch_handle_calls = 0
        self.forward_features: list[object] = []
        self.reset_perf_stats_calls = 0
        self.fast_path_mode = "auto"
        _FakeRecStoreEmbeddingBagCollection.last_instance = self

    def issue_fused_prefetch(self, features, *, record_handle: bool = True):
        self.issue_fused_prefetch_features.append(features)
        self.issue_fused_prefetch_calls += 1
        self.issue_fused_prefetch_record_flags.append(bool(record_handle))
        return (
            1000 + self.issue_fused_prefetch_calls,
            7,
            12.5,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    def issue_fused_id_prefetch(self, fused_ids, *, record_handle: bool = True):
        self.issue_fused_id_prefetch_ids.append(fused_ids.detach().cpu().clone())
        self.issue_fused_prefetch_calls += 1
        self.issue_fused_prefetch_record_flags.append(bool(record_handle))
        return (
            2000 + self.issue_fused_prefetch_calls,
            int(fused_ids.numel()),
            12.5,
            fused_ids.detach().to(dtype=torch.int64, device="cpu").flatten(),
            torch.arange(int(fused_ids.numel()), dtype=torch.int64),
        )

    def prepare_fused_prefetch(self, features):
        self.prepare_fused_prefetch_calls += 1
        self.issue_fused_prefetch_features.append(features)
        return (
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            7,
        )

    def issue_prepared_fused_prefetch(
        self, unique_ids, inverse, num_ids, *, record_handle: bool = True
    ):
        del unique_ids, inverse
        self.issue_fused_prefetch_calls += 1
        self.issue_fused_prefetch_record_flags.append(bool(record_handle))
        return 3000 + self.issue_fused_prefetch_calls

    def set_fused_prefetch_handle(self, *args, **kwargs) -> None:
        del args, kwargs
        self.set_fused_prefetch_handle_calls += 1

    def reset_perf_stats(self) -> None:
        self.reset_perf_stats_calls += 1

    def consume_perf_stats(self, reset: bool = True):
        del reset
        return {
            "lookup_ids_build_ms": 0.1,
            "lookup_wait_ms": 0.6,
            "lookup_total_ms": 0.9,
        }

    def __call__(self, features):
        self.forward_features.append(features)
        return object()

    def resolve_fast_path_backend(self):
        return None


class _FakePrefetchModule:
    def __init__(self) -> None:
        self.next_handle = 100
        self.issued: list[tuple[object, bool]] = []
        self.issued_fused_ids: list[torch.Tensor] = []
        self.consumed: list[tuple[int, int]] = []
        self.consume_kwargs: list[dict[str, object]] = []

    def issue_fused_prefetch(self, features, *, record_handle: bool = True):
        self.issued.append((features, record_handle))
        handle = self.next_handle
        self.next_handle += 1
        return (
            handle,
            int(getattr(features, "num_ids", 0)),
            10.0 + handle,
            torch.tensor([handle], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    def issue_fused_id_prefetch(self, fused_ids, *, record_handle: bool = True):
        del record_handle
        self.issued_fused_ids.append(fused_ids.detach().cpu().clone())
        handle = self.next_handle
        self.next_handle += 1
        fused_ids_cpu = fused_ids.detach().to(dtype=torch.int64, device="cpu").flatten()
        return (
            handle,
            int(fused_ids_cpu.numel()),
            10.0 + handle,
            fused_ids_cpu,
            torch.arange(int(fused_ids_cpu.numel()), dtype=torch.int64),
        )

    def set_fused_prefetch_handle(self, handle, num_ids=None, issue_ts=None, **kwargs) -> None:
        del issue_ts
        self.consumed.append((int(handle), int(num_ids or 0)))
        self.consume_kwargs.append(dict(kwargs))


class _FakeSparseFeatures:
    def __init__(self, num_ids: int) -> None:
        self.num_ids = int(num_ids)


class _FakeJaggedFeature:
    def __init__(self, values: torch.Tensor) -> None:
        self._values = values

    def values(self) -> torch.Tensor:
        return self._values


class _FakeKeyedSparseFeatures:
    def __init__(self, values: list[int]) -> None:
        self._values = torch.tensor(values, dtype=torch.int64)

    def keys(self):
        return ["cat_0"]

    def __getitem__(self, key: str) -> _FakeJaggedFeature:
        if key != "cat_0":
            raise KeyError(key)
        return _FakeJaggedFeature(self._values)


class _FakeSparseSGD:
    last_instance = None

    def __init__(self, params, lr: float) -> None:
        self.params = params
        self.lr = lr
        self.step_calls = 0
        self.flush_calls = 0
        self.zero_grad_calls = 0
        self.reset_perf_stats_calls = 0
        self._last_update_payloads = []
        _FakeSparseSGD.last_instance = self

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1
        self._last_update_payloads = []
        for mod in self.params:
            forward_features = getattr(mod, "forward_features", [])
            if not forward_features:
                continue
            feature = forward_features[-1]
            values = getattr(feature, "_values", torch.empty((0,), dtype=torch.int64))
            ids = values.detach().to(dtype=torch.int64)
            if ids.numel() == 0:
                continue
            self._last_update_payloads.append(
                {
                    "module": mod,
                    "name": "t0",
                    "ids": ids,
                    "grads": torch.ones((int(ids.numel()), 4), dtype=torch.float32),
                    "lr": float(self.lr),
                }
            )

    def flush(self):
        self.flush_calls += 1

    def reset_perf_stats(self) -> None:
        self.reset_perf_stats_calls += 1

    def consume_perf_stats(self, reset: bool = True):
        del reset
        return {
        }

    def last_update_payloads(self):
        return list(self._last_update_payloads)


class _FakeDenseOptimizer:
    def __init__(self, params, lr: float) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs) -> None:
        del args, kwargs
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class TestRecStoreRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._append_worker_debug_patch = mock.patch(
            "model_zoo.rs_demo.runners.recstore_runner._append_worker_debug",
            lambda *args, **kwargs: None,
        )
        self._append_worker_debug_patch.start()
        self.addCleanup(self._append_worker_debug_patch.stop)

    def test_lookahead_prefetcher_depth_zero_never_issues_prefetch(self) -> None:
        module = _FakePrefetchModule()
        prefetcher = LookaheadPrefetcher(module, depth=0, embedding_dim=128)

        prefetcher.enqueue(_FakeSparseFeatures(3))
        prefetcher.attach_next()

        self.assertEqual(module.issued, [])
        self.assertEqual(module.consumed, [])

    def test_lookahead_prefetcher_delays_consumption_by_depth(self) -> None:
        module = _FakePrefetchModule()
        prefetcher = LookaheadPrefetcher(module, depth=2, embedding_dim=128)

        prefetcher.enqueue(_FakeSparseFeatures(3))
        self.assertFalse(prefetcher.attach_next())

        prefetcher.enqueue(_FakeSparseFeatures(5))
        self.assertFalse(prefetcher.attach_next())

        prefetcher.enqueue(_FakeSparseFeatures(7))
        self.assertTrue(prefetcher.advance())
        self.assertTrue(prefetcher.attach_next())

        self.assertEqual([item[1] for item in module.issued], [False, False, False])
        self.assertEqual(module.consumed, [(100, 3)])
        self.assertEqual(prefetcher.live_ids, 12)
        self.assertEqual(prefetcher.live_bytes, 12 * 128 * 4)

    def test_lookahead_prefetcher_can_discard_stale_ready_handle(self) -> None:
        module = _FakePrefetchModule()
        prefetcher = LookaheadPrefetcher(module, depth=1, embedding_dim=64)

        prefetcher.enqueue(_FakeSparseFeatures(10))
        prefetcher.enqueue(_FakeSparseFeatures(20))
        self.assertTrue(prefetcher.advance())
        self.assertTrue(prefetcher.discard_next_ready())
        self.assertFalse(prefetcher.attach_next())

        prefetcher.enqueue(_FakeSparseFeatures(30))
        self.assertTrue(prefetcher.advance())
        self.assertTrue(prefetcher.attach_next())

        self.assertEqual(module.consumed, [(101, 20)])

    def test_lookahead_prefetcher_can_enqueue_bagpipe_fused_ids(self) -> None:
        module = _FakePrefetchModule()
        prefetcher = LookaheadPrefetcher(module, depth=1, embedding_dim=64)

        prefetcher.enqueue_fused_ids(torch.tensor([7, 8], dtype=torch.int64))
        prefetcher.enqueue_fused_ids(torch.tensor([9], dtype=torch.int64))
        self.assertTrue(prefetcher.advance())
        self.assertTrue(prefetcher.attach_next())

        self.assertEqual([ids.tolist() for ids in module.issued_fused_ids], [[7, 8], [9]])
        self.assertEqual(module.consumed, [(100, 2)])

    def test_finalize_step_timing_records_total_throughput(self) -> None:
        row = {"batch_size": 64}
        with mock.patch(
            "model_zoo.rs_demo.runners.recstore_runner.time.perf_counter",
            return_value=12.0,
        ):
            recstore_runner._finalize_step_timing(
                row, consume_start=10.0, wall_start=9.0
            )

        self.assertEqual(row["step_total_ms"], 3000.0)
        self.assertAlmostEqual(row["samples_per_sec"], 64 / 3)
        self.assertAlmostEqual(row["batches_per_sec"], 1 / 3)
        self.assertEqual(row["step_end_to_end_ms"], 3000.0)

    def test_warmup_gpu_local_shm_fast_path_runs_only_for_shared_cuda_fast_path(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            single_node_ps_backend="local_shm",
        )
        client = _FakeRecStoreClient()
        client._shared_local_shm_table = True

        warmed = recstore_runner._maybe_warmup_gpu_local_shm_fast_path(
            cfg=cfg,
            client=client,
            device=torch.device("cuda:0"),
        )

        self.assertTrue(warmed)
        self.assertEqual(client.local_shm_warmup_calls, 1)

    def test_warmup_gpu_local_shm_fast_path_skips_hierkv_backend(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            single_node_ps_backend="hierkv",
        )
        client = _FakeRecStoreClient()
        client._shared_local_shm_table = True
        client.set_ps_backend("hierkv")

        warmed = recstore_runner._maybe_warmup_gpu_local_shm_fast_path(
            cfg=cfg,
            client=client,
            device=torch.device("cuda:0"),
        )

        self.assertFalse(warmed)
        self.assertEqual(client.current_ps_backend(), "hierkv")
        self.assertEqual(client.local_shm_warmup_calls, 0)

    def test_warmup_gpu_local_shm_fast_path_skips_when_conditions_do_not_match(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=2,
            single_node_ps_backend="local_shm",
        )
        client = _FakeRecStoreClient()
        client._shared_local_shm_table = True

        warmed = recstore_runner._maybe_warmup_gpu_local_shm_fast_path(
            cfg=cfg,
            client=client,
            device=torch.device("cuda:0"),
        )

        self.assertFalse(warmed)
        self.assertEqual(client.local_shm_warmup_calls, 0)

    def _run_local_worker_with_fake_embedding_module(
        self,
        cfg: RunConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        num_batches: int = 1,
        build_kjt=None,
        captured_rows: list | None = None,
        write_csv: bool = False,
        use_run: bool = False,
        patch_distributed: bool = False,
    ) -> _FakeRecStoreEmbeddingBagCollection:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)] * num_batches
        dataloader = list(dataset)
        if build_kjt is None:
            build_kjt = lambda *args, **kwargs: (None, object())

        fake_client = _FakeDirectReadRecStoreClient()
        fake_embeddingbag_module = types.ModuleType("python.pytorch.torchrec_kv.EmbeddingBag")
        class _ProfiledEmbeddingBagCollection(_FakeRecStoreEmbeddingBagCollection):
            def consume_perf_stats(self, reset: bool = True):
                del reset
                return {
                    "lookup_ids_build_ms": 0.5,
                    "lookup_wait_ms": 0.75,
                    "lookup_total_ms": 1.25,
                }

        fake_embeddingbag_module.RecStoreEmbeddingBagCollection = _ProfiledEmbeddingBagCollection
        fake_optimizer_module = types.ModuleType("python.pytorch.recstore.optimizer")
        fake_optimizer_module.SparseSGD = _FakeSparseSGD

        _FakeRecStoreEmbeddingBagCollection.last_instance = None

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    "sys.modules",
                    {
                        "python.pytorch.torchrec_kv.EmbeddingBag": fake_embeddingbag_module,
                        "python.pytorch.recstore.optimizer": fake_optimizer_module,
                        "python.pytorch.recstore.KVClient": types.SimpleNamespace(RecStoreClient=lambda: fake_client),
                    },
                )
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None)
            )
            stack.enter_context(
                mock.patch("model_zoo.rs_demo.runners.recstore_runner.torch.manual_seed", lambda *_: None)
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.torch.optim.SGD",
                    _FakeDenseOptimizer,
                )
            )
            stack.enter_context(
                mock.patch.object(recstore_runner.recstore, "RecStoreClient", return_value=fake_client)
            )
            stack.enter_context(
                mock.patch.multiple(recstore_runner.recstore, SparseSGD=fake_optimizer_module.SparseSGD, RecStoreEmbeddingBagCollection=fake_embeddingbag_module.RecStoreEmbeddingBagCollection)
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                    lambda: ["cat_0"],
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                    lambda **kwargs: (dataset, dataloader),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                    build_kjt,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.build_dense_module",
                    lambda *args, **kwargs: _DummyDense().to(kwargs["device"]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.reshape_torchrec_embeddings_for_dlrm",
                    lambda **kwargs: torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.prepare_hybrid_dlrm_input",
                    lambda **kwargs: (
                        torch.zeros((1, 13), dtype=torch.float32, device=kwargs["device"]),
                        torch.zeros((1, 1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                        torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                    ),
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                    lambda row: row,
                )
            )
            stack.enter_context(
                mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                    lambda xs: "ok",
                )
            )
            if captured_rows is not None:
                stack.enter_context(
                    mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner._write_rows",
                        lambda path, rows: captured_rows.extend(rows),
                    )
                )
            elif not write_csv:
                stack.enter_context(
                    mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner._write_rows",
                        lambda *args, **kwargs: None,
                    )
                )
            if patch_distributed:
                stack.enter_context(
                    mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner._maybe_wrap_dense_module_for_dist",
                        lambda **kwargs: kwargs["dense_module"],
                    )
                )
                fake_dist = types.SimpleNamespace(
                    is_initialized=lambda: False,
                    init_process_group=lambda **kwargs: None,
                    barrier=lambda *args, **kwargs: None,
                    destroy_process_group=lambda: None,
                )
                stack.enter_context(mock.patch("torch.distributed.is_initialized", fake_dist.is_initialized))
                stack.enter_context(mock.patch("torch.distributed.init_process_group", fake_dist.init_process_group))
                stack.enter_context(mock.patch("torch.distributed.barrier", fake_dist.barrier))
                stack.enter_context(mock.patch("torch.distributed.destroy_process_group", fake_dist.destroy_process_group))

            runner = RecStoreRunner(runner_runtime)
            if use_run:
                runner.run(repo_root=repo_root, cfg=cfg)
            else:
                runner._run_local_worker(
                    repo_root=repo_root,
                    cfg=cfg,
                    rank=rank,
                    world_size=world_size,
                    local_rank=local_rank,
                    out_csv=runner_runtime / "rank.csv",
                )

        fake_ebc = _FakeRecStoreEmbeddingBagCollection.last_instance
        self.assertIsNotNone(fake_ebc)
        return fake_ebc

    def test_parse_config_defaults_single_node_ps_backend(self) -> None:
        cfg = config.parse_config(["--backend", "recstore"])

        self.assertEqual(cfg.nnodes, 1)
        self.assertEqual(cfg.single_node_ps_backend, "local_shm")
        self.assertEqual(cfg.single_node_owner_policy, "hash_mod_world_size")

    def test_parse_config_json_drops_removed_enable_single_node_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_worker.json"
            path.write_text(
                '{"backend": "recstore", "nnodes": 1, '
                '"enable_single_node_distributed_fast_path": true, '
                '"single_node_ps_backend": "hierkv"}',
                encoding="utf-8",
            )
            loaded = config.parse_config(["--run-config-json", str(path)])
        self.assertEqual(loaded.single_node_ps_backend, "hierkv")
        self.assertFalse(hasattr(loaded, "enable_single_node_distributed_fast_path"))

    def test_parse_config_accepts_gpu_cache_options(self) -> None:
        cfg = config.parse_config(
            [
                "--backend",
                "recstore",
                "--enable-gpu-cache",
                "--gpu-cache-capacity",
                "1024",
                "--disable-gpu-cache-lookup-bypass",
            ]
        )

        self.assertTrue(cfg.enable_gpu_cache)
        self.assertEqual(cfg.gpu_cache_capacity, 1024)
        self.assertTrue(cfg.disable_gpu_cache_lookup_bypass)

    def test_parse_config_accepts_read_mode_bagpipe_choice(self) -> None:
        cfg = config.parse_config(["--backend", "recstore", "--read-mode", "bagpipe"])
        self.assertEqual(cfg.read_mode, "bagpipe")

    def test_parse_config_accepts_prefetch_depth(self) -> None:
        cfg = config.parse_config(
            [
                "--backend",
                "recstore",
                "--prefetch-depth",
                "4",
            ]
        )

        self.assertEqual(cfg.prefetch_depth, 4)

    def test_parse_config_accepts_prefetch_issue_depth(self) -> None:
        cfg = config.parse_config(
            [
                "--backend",
                "recstore",
                "--prefetch-issue-depth",
                "12",
            ]
        )

        self.assertEqual(cfg.prefetch_issue_depth, 12)

    def test_validate_recstore_config_auto_sets_bagpipe_for_gpu_cache(self) -> None:
        cfg = RunConfig(backend="recstore", enable_gpu_cache=True, gpu_cache_capacity=1024)

        config.validate_recstore_config(cfg)
        self.assertEqual(cfg.optimization.plugin, "bagpipe")

    def test_validate_recstore_config_auto_sets_bagpipe_plugin(self) -> None:
        cfg = RunConfig(backend="recstore", read_mode="bagpipe")

        config.validate_recstore_config(cfg)
        self.assertEqual(cfg.optimization.plugin, "bagpipe")
        self.assertGreater(cfg.optimization.lookahead, 0)

    def test_validate_recstore_config_rejects_negative_prefetch_depth(self) -> None:
        cfg = RunConfig(backend="recstore", prefetch_depth=-1)

        with self.assertRaisesRegex(RuntimeError, "--prefetch-depth must be non-negative"):
            config.validate_recstore_config(cfg)

    def test_validate_recstore_config_rejects_negative_prefetch_issue_depth(self) -> None:
        cfg = RunConfig(backend="recstore", prefetch_issue_depth=-1)

        with self.assertRaisesRegex(RuntimeError, "--prefetch-issue-depth must be non-negative"):
            config.validate_recstore_config(cfg)

    def test_validate_recstore_config_allows_single_node_ps_backend(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=2,
            single_node_ps_backend="local_shm",
            single_node_owner_policy="hash_mod_world_size",
        )

        config.validate_recstore_config(cfg)

    def test_validate_recstore_config_allows_hierkv_single_node_ps_backend(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=2,
            single_node_ps_backend="hierkv",
            single_node_owner_policy="hash_mod_world_size",
        )

        config.validate_recstore_config(cfg)

    def test_validate_recstore_config_allows_single_node_with_one_process(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=1,
            single_node_ps_backend="local_shm",
        )

        config.validate_recstore_config(cfg)

    def test_validate_recstore_config_skips_single_node_ps_checks_for_multi_node(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=2,
            nproc_per_node=2,
            node_rank=0,
            recstore_runtime_dir="/tmp/shared",
            single_node_ps_backend="brpc",
            single_node_owner_policy="rank_zero",
        )

        config.validate_recstore_config(cfg)

    def test_validate_recstore_config_rejects_invalid_fast_path_backend(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=2,
            single_node_ps_backend="brpc",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "single-node path only supports --single-node-ps-backend=local_shm or hierkv",
        ):
            config.validate_recstore_config(cfg)

    def test_parse_config_rejects_invalid_single_node_owner_policy_choice(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                config.parse_config(
                    [
                        "--single-node-owner-policy",
                        "invalid_policy",
                    ]
                )

    def test_parse_config_rejects_invalid_single_node_ps_backend_choice(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                config.parse_config(
                    [
                        "--single-node-ps-backend",
                        "invalid_backend",
                    ]
                )

    def test_wrap_dense_module_for_dist_uses_ddp_when_distributed(self) -> None:
        module = _DummyDense()
        wrapped = object()

        with mock.patch(
            "torch.nn.parallel.DistributedDataParallel",
            return_value=wrapped,
        ) as ddp_ctor:
            result = _maybe_wrap_dense_module_for_dist(
                dense_module=module,
                device=torch.device("cpu"),
                local_rank=0,
                use_dist=True,
            )

        self.assertIs(result, wrapped)
        ddp_ctor.assert_called_once_with(module)

    def test_build_train_dataloader_for_distributed_uses_rank_partition(self) -> None:
        fake_dataset = [1, 2, 3]

        with mock.patch(
            "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
            return_value=(fake_dataset, "loader"),
        ) as build_loader:
            dataset, dataloader = _build_train_dataloader_for_mode(
                repo_root=Path("/app/RecStore"),
                cfg=RunConfig(
                    backend="recstore",
                    steps=1,
                    nnodes=2,
                    nproc_per_node=1,
                    batch_size=256,
                ),
                rank=1,
            )

        self.assertEqual(dataset, fake_dataset)
        self.assertEqual(dataloader, "loader")
        self.assertEqual(build_loader.call_args.kwargs["seed"], 20260330)
        self.assertEqual(build_loader.call_args.kwargs["shuffle"], True)
        self.assertEqual(build_loader.call_args.kwargs["rank"], 1)
        self.assertEqual(build_loader.call_args.kwargs["world_size"], 2)

    def test_runner_uses_world_size_from_nnodes_and_nproc_per_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            runner = RecStoreRunner(runtime_dir=runtime_dir)
            cfg = config.RunConfig(
                backend="recstore",
                nnodes=1,
                node_rank=0,
                nproc_per_node=2,
                output_root=tmpdir,
                run_id="recstore-dist",
            )

            with mock.patch.object(
                runner,
                "_run_distributed",
                return_value={"backend": "recstore", "rows": []},
            ) as dist_run:
                result = runner.run(Path(tmpdir), cfg)
            dist_run.assert_called_once_with(Path(tmpdir), cfg)

    def test_build_torchrun_cmd_uses_config_json(self) -> None:
        runner = RecStoreRunner(Path("/tmp/runtime"))
        cfg = RunConfig(
            backend="recstore", nnodes=1, node_rank=0, nproc_per_node=2,
            master_addr="127.0.0.1", master_port=29653, rdzv_backend="c10d",
            rdzv_id="recstore-case", output_root="/tmp/rs_demo", run_id="recstore-case",
        )
        cmd = runner._build_torchrun_cmd(
            Path("/app/RecStore"), cfg, Path("/tmp/worker_config.json")
        )
        self.assertIn("--run-config-json", cmd)
        self.assertIn("/tmp/worker_config.json", cmd)
        self.assertIn("--nproc_per_node", cmd)
        self.assertIn("2", cmd)

    def test_local_worker_rows_include_ps_kv_backend_label(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            ps_kv_backend="hps_rocksdb",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="direct",
            recstore_main_csv="/tmp/recstore-ps-kv.csv",
        )

        captured_rows: list[dict] = []
        self._run_local_worker_with_fake_embedding_module(cfg, captured_rows=captured_rows)

        self.assertEqual(len(captured_rows), 1)
        self.assertNotIn("backend", captured_rows[0])
        self.assertNotIn("ps_kv_backend", captured_rows[0])
        self.assertNotIn("model_backend_label", captured_rows[0])

    def test_embedding_module_auto_detects_fast_path_without_runner_injection(self) -> None:
        # Neither the default path nor the explicit single-node config should make
        # the runner inject a fast-path mode; the module keeps auto-detecting.
        base_kwargs = dict(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
        )
        for extra in (
            {"recstore_main_csv": "/tmp/recstore-default.csv"},
            {
                "nnodes": 1,
                "nproc_per_node": 2,
                "single_node_ps_backend": "local_shm",
                "single_node_owner_policy": "hash_mod_world_size",
                "recstore_main_csv": "/tmp/recstore-fast-path.csv",
            },
        ):
            cfg = RunConfig(**base_kwargs, **extra)
            fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg)
            self.assertEqual(fake_ebc.fast_path_mode, "auto")

    def test_gpu_cache_options_auto_set_bagpipe_via_validate(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            enable_gpu_cache=True,
            gpu_cache_capacity=1024,
            disable_gpu_cache_lookup_bypass=True,
        )
        config.validate_recstore_config(cfg)
        self.assertEqual(cfg.optimization.plugin, "bagpipe")

    def test_local_worker_switches_client_backend_for_single_node_fast_path(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            nnodes=1,
            nproc_per_node=2,
            single_node_ps_backend="hierkv",
            single_node_owner_policy="hash_mod_world_size",
            recstore_main_csv="/tmp/recstore-hierkv-fast-path.csv",
        )

        fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg)

        self.assertEqual(fake_ebc.kv_client.set_ps_backend_calls, ["hierkv"])

    def test_prefetch_mode_uses_ebc_prefetch_and_sparse_optimizer(self) -> None:
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="prefetch",
            prefetch_depth=0,
            recstore_main_csv=str(Path(tempfile.mkdtemp()) / "main.csv"),
        )

        fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg, use_run=True)
        fake_sparse_optimizer = _FakeSparseSGD.last_instance
        self.assertIsNotNone(fake_ebc)
        self.assertIsNotNone(fake_sparse_optimizer)
        self.assertEqual(fake_ebc.issue_fused_prefetch_calls, 1)
        self.assertEqual(fake_ebc.issue_fused_prefetch_record_flags, [True])
        self.assertIs(fake_ebc.kwargs["kv_client"], fake_ebc.kv_client)
        self.assertEqual(fake_ebc.kv_client.emb_read_prefetch_calls, 0)
        self.assertEqual(fake_sparse_optimizer.step_calls, 1)
        self.assertEqual(fake_sparse_optimizer.flush_calls, 1)
        self.assertGreaterEqual(fake_sparse_optimizer.zero_grad_calls, 2)
        self.assertEqual(fake_ebc.reset_perf_stats_calls, 1)

    def test_direct_mode_skips_async_issue(self) -> None:
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="direct",
            recstore_main_csv=str(Path(tempfile.mkdtemp()) / "main.csv"),
        )
        fake_ebc = self._run_local_worker_with_fake_embedding_module(cfg, use_run=True)
        self.assertEqual(fake_ebc.issue_fused_prefetch_calls, 0)
        self.assertEqual(fake_ebc.set_fused_prefetch_handle_calls, 0)

    def test_prefetch_depth_uses_lookahead_handles(self) -> None:
        cfg = RunConfig(
            steps=3,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="prefetch",
            prefetch_depth=1,
            recstore_main_csv=str(Path(tempfile.mkdtemp()) / "main.csv"),
        )

        feature_values_by_build = [4, 5, 4]
        built_sparse_features: list[object] = []
        build_devices: list[torch.device | None] = []
        build_device_by_feature: dict[object, torch.device | None] = {}

        def build_fake_kjt(*args, **kwargs):
            feature = _FakeKeyedSparseFeatures(
                [feature_values_by_build[len(built_sparse_features)]]
            )
            built_sparse_features.append(feature)
            build_device = kwargs.get("device")
            build_devices.append(build_device)
            build_device_by_feature[feature] = build_device
            return None, feature

        captured_rows: list[dict] = []
        fake_ebc = self._run_local_worker_with_fake_embedding_module(
            cfg,
            use_run=True,
            num_batches=3,
            build_kjt=build_fake_kjt,
            captured_rows=captured_rows,
        )
        self.assertEqual(
            fake_ebc.issue_fused_prefetch_record_flags,
            [False, False, False],
        )
        self.assertEqual(fake_ebc.set_fused_prefetch_handle_calls, 3)
        self.assertEqual(len(built_sparse_features), 3)
        expected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.assertEqual([str(device) for device in build_devices], [expected_device] * 3)
        self.assertEqual(len(fake_ebc.issue_fused_prefetch_features), 3)
        self.assertEqual(fake_ebc.issue_fused_id_prefetch_ids, [])
        self.assertEqual(len(fake_ebc.forward_features), 3)
        self.assertEqual(
            [str(build_device_by_feature[feature]) for feature in fake_ebc.forward_features],
            [expected_device, expected_device, expected_device],
        )
        self.assertTrue(
            all(row["step_end_to_end_ms"] >= row["step_total_ms"] for row in captured_rows)
        )
        self.assertTrue(
            all(row["step_end_to_end_ms"] == row["step_total_ms"] for row in captured_rows)
        )

    def test_local_worker_emits_perf_breakdown_columns_from_model_layer_stats(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="prefetch",
            recstore_main_csv=str(Path(tempfile.mkdtemp()) / "main.csv"),
        )

        captured_rows: list[dict] = []
        self._run_local_worker_with_fake_embedding_module(
            cfg, use_run=True, captured_rows=captured_rows
        )

        self.assertEqual(len(captured_rows), 1)
        row = captured_rows[0]
        self.assertIn("lookup_wait_ms", row)
        self.assertIn("lookup_ids_build_ms", row)
        self.assertIn("lookup_total_ms", row)
        self.assertNotIn("prefetch_issue_ms", row)
        self.assertNotIn("update_owner_exchange_ms", row)

    def test_runner_exports_coarse_perf_stats_into_rows(self) -> None:
        csv_path = Path(tempfile.mkdtemp()) / "main.csv"
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="direct",
            recstore_main_csv=str(csv_path),
        )

        self._run_local_worker_with_fake_embedding_module(cfg, use_run=True, write_csv=True)

        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertTrue(math.isfinite(float(rows[0]["loss"])))
        self.assertIn("lookup_wait_ms", rows[0])
        self.assertIn("sparse_backward_replay_ms", rows[0])
        self.assertIn("sparse_optimizer_step_ms", rows[0])
        self.assertIn("sparse_optimizer_flush_ms", rows[0])
        self.assertNotIn("sparse_zero_grad_ms", rows[0])
        self.assertNotIn("update_owner_exchange_ms", rows[0])
        self.assertNotIn("lookup_local_lookup_ms", rows[0])

    def test_runner_passes_compute_device_into_sparse_feature_builder(self) -> None:
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="direct",
            recstore_main_csv=str(Path(tempfile.mkdtemp()) / "main.csv"),
        )

        device_calls: list[torch.device] = []

        def _build_sparse_features(*args, **kwargs):
            device_calls.append(kwargs["device"])
            return None, object()

        self._run_local_worker_with_fake_embedding_module(
            cfg, use_run=True, build_kjt=_build_sparse_features
        )

        self.assertGreaterEqual(len(device_calls), 1)
        expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertTrue(all(device.type == expected_device_type for device in device_calls))

    def test_nonzero_rank_skips_table_init_and_warm_write(self) -> None:
        output_root = Path(tempfile.mkdtemp())
        cfg = RunConfig(
            backend="recstore",
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_mode="direct",
            nnodes=1,
            nproc_per_node=2,
            output_root=str(output_root),
            recstore_main_csv=str(output_root / "main.csv"),
        )

        fake_ebc = self._run_local_worker_with_fake_embedding_module(
            cfg, rank=1, world_size=2, local_rank=0, patch_distributed=True
        )

        self.assertFalse(fake_ebc.kwargs["initialize_tables"])
        self.assertEqual(fake_ebc.kv_client.emb_write_calls, 0)

    def test_merge_rank_outputs_preserves_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rank0 = Path(tmpdir) / "rank0.csv"
            rank1 = Path(tmpdir) / "rank1.csv"
            out_csv = Path(tmpdir) / "main.csv"
            recstore_runner._write_rows(
                rank1,
                [
                    {
                        "backend": "recstore",
                        "dist_mode": "single_node",
                        "rank": 1,
                        "step": 0,
                        "step_total_ms": 11.0,
                    }
                ],
            )
            recstore_runner._write_rows(
                rank0,
                [
                    {
                        "backend": "recstore",
                        "dist_mode": "single_node",
                        "rank": 0,
                        "step": 1,
                        "step_total_ms": 9.0,
                    },
                    {
                        "backend": "recstore",
                        "dist_mode": "single_node",
                        "rank": 0,
                        "step": 0,
                        "step_total_ms": 10.0,
                    },
                ],
            )

            rows = recstore_runner._merge_rank_outputs([rank1, rank0], out_csv)

            self.assertEqual([(row["rank"], row["step"]) for row in rows], [(0, 0), (0, 1), (1, 0)])
            self.assertTrue(out_csv.exists())


if __name__ == "__main__":
    unittest.main()
