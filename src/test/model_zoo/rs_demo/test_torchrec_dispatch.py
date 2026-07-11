from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest import mock
import unittest
import torch

from model_zoo.rs_demo.cli import build_runner
from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runtime.report import write_stage_csv
from model_zoo.rs_demo.runners.torchrec_runner import (
    TorchRecRunner,
    _barrier_for_step_alignment,
    _build_worker_fingerprint,
    _build_uvm_caching_constraints,
    _merge_rank_outputs,
    _debug_log_path,
    _maybe_wrap_dense_module_for_dist,
    _parse_nccl_transport_log,
    _write_or_verify_worker_fingerprint,
    _compute_or_load_shared_sharding_plan,
    _summarize_sharding_plan,
    _build_train_dataloader_for_mode,
)


class _FakeDist:
    def __init__(self) -> None:
        self.barrier_calls = 0

    def barrier(self, device_ids=None) -> None:
        self.barrier_calls += 1


class _FakePlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, module, sharders):
        self.calls += 1
        return {"module": module, "sharders": sharders, "token": "rank0-plan"}


class _FakeParameterSharding:
    def __init__(self, sharding_type: str, compute_kernel: str, ranks: list[int]) -> None:
        self.sharding_type = sharding_type
        self.compute_kernel = compute_kernel
        self.ranks = ranks


class _FakeEmbeddingComputeKernelValue:
    value = "fused_uvm_caching"


class _FakeEmbeddingComputeKernel:
    FUSED_UVM_CACHING = _FakeEmbeddingComputeKernelValue()


class _FakeParameterConstraints:
    def __init__(self, *, compute_kernels):
        self.compute_kernels = compute_kernels


class _FakePlan:
    def __init__(self, plan_by_module):
        self.plan = plan_by_module


class TestTorchRecDispatch(unittest.TestCase):
    def test_debug_log_path_is_rank_scoped(self) -> None:
        cfg = RunConfig(output_root="/tmp/rs_demo", run_id="case-debug")

        path = _debug_log_path(cfg, rank=7)

        self.assertEqual(
            path,
            Path("/tmp/rs_demo/outputs/case-debug/torchrec_worker_rank7.log"),
        )

    def test_build_worker_fingerprint_includes_critical_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            for rel_path in (
                "model_zoo/rs_demo/config.py",
                "model_zoo/rs_demo/data/dlrm_source.py",
                "model_zoo/rs_demo/runners/torchrec_runner.py",
                "model_zoo/rs_demo/runtime/hybrid_dlrm.py",
            ):
                path = repo_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rel_path, encoding="utf-8")

            fingerprint = _build_worker_fingerprint(repo_root)

        self.assertIn("files", fingerprint)
        self.assertIn("model_zoo/rs_demo/config.py", fingerprint["files"])
        self.assertIn("model_zoo/rs_demo/data/dlrm_source.py", fingerprint["files"])
        self.assertIn("model_zoo/rs_demo/runners/torchrec_runner.py", fingerprint["files"])
        self.assertIn("model_zoo/rs_demo/runtime/hybrid_dlrm.py", fingerprint["files"])

    def test_write_or_verify_worker_fingerprint_accepts_matching_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fingerprint_path = Path(tmpdir) / "fingerprints.json"
            fingerprint = {"files": {"a.py": "hash-a"}}

            _write_or_verify_worker_fingerprint(
                rank=0,
                world_size=2,
                fingerprint=fingerprint,
                fingerprint_path=fingerprint_path,
            )
            _write_or_verify_worker_fingerprint(
                rank=1,
                world_size=2,
                fingerprint=fingerprint,
                fingerprint_path=fingerprint_path,
            )

            stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))

        self.assertEqual(stored["0"], fingerprint)
        self.assertEqual(stored["1"], fingerprint)

    def test_write_or_verify_worker_fingerprint_tolerates_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fingerprint_path = Path(tmpdir) / "fingerprints.json"
            fingerprint_path.write_text("", encoding="utf-8")
            fingerprint = {"files": {"a.py": "hash-a"}}

            _write_or_verify_worker_fingerprint(
                rank=0,
                world_size=2,
                fingerprint=fingerprint,
                fingerprint_path=fingerprint_path,
            )

            stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))

        self.assertEqual(stored["0"], fingerprint)

    def test_write_or_verify_worker_fingerprint_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fingerprint_path = Path(tmpdir) / "fingerprints.json"
            _write_or_verify_worker_fingerprint(
                rank=0,
                world_size=2,
                fingerprint={"files": {"a.py": "hash-a"}},
                fingerprint_path=fingerprint_path,
            )

            with self.assertRaisesRegex(RuntimeError, "worker fingerprint mismatch"):
                _write_or_verify_worker_fingerprint(
                    rank=1,
                    world_size=2,
                    fingerprint={"files": {"a.py": "hash-b"}},
                    fingerprint_path=fingerprint_path,
                )

    def test_barrier_for_step_alignment_uses_local_rank_on_cuda(self) -> None:
        fake_dist = _FakeDist()
        device = mock.Mock()
        device.type = "cuda"

        _barrier_for_step_alignment(fake_dist, device, local_rank=3, use_dist=True)

        self.assertEqual(fake_dist.barrier_calls, 1)

    def test_barrier_for_step_alignment_skips_single_process(self) -> None:
        fake_dist = _FakeDist()
        device = mock.Mock()
        device.type = "cpu"

        _barrier_for_step_alignment(fake_dist, device, local_rank=0, use_dist=False)

        self.assertEqual(fake_dist.barrier_calls, 0)

    def test_plan_summary_formats_table_entries(self) -> None:
        summary = _summarize_sharding_plan(
            _FakePlan(
                {
                    "": {
                        "t_cat_0": _FakeParameterSharding("table_wise", "dense", [0]),
                        "t_cat_1": _FakeParameterSharding("row_wise", "fused", [1]),
                    }
                }
            )
        )

        self.assertIn("module=<root>", summary)
        self.assertIn("t_cat_0:table_wise:dense:ranks=[0]", summary)
        self.assertIn("t_cat_1:row_wise:fused:ranks=[1]", summary)

    def test_rank0_computes_and_persists_shared_sharding_plan(self) -> None:
        fake_dist = _FakeDist()
        planner = _FakePlanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = _compute_or_load_shared_sharding_plan(
                dist=fake_dist,
                rank=0,
                embedding_module="emb",
                sharders=["s"],
                planner=planner,
                plan_path=Path(tmpdir) / "plan.pkl",
            )

        self.assertEqual(plan["token"], "rank0-plan")
        self.assertEqual(planner.calls, 1)
        self.assertEqual(fake_dist.barrier_calls, 0)

    def test_nonzero_rank_loads_shared_sharding_plan(self) -> None:
        fake_dist = _FakeDist()
        planner = _FakePlanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.pkl"
            with plan_path.open("wb") as f:
                pickle.dump({"token": "rank0-plan"}, f)
            plan = _compute_or_load_shared_sharding_plan(
                dist=fake_dist,
                rank=1,
                embedding_module="emb",
                sharders=["s"],
                planner=planner,
                plan_path=plan_path,
            )

        self.assertEqual(plan["token"], "rank0-plan")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(fake_dist.barrier_calls, 0)

    def test_build_runner_torchrec_requires_dependency(self) -> None:
        cfg = RunConfig(backend="torchrec")
        with mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_torchrec_available",
            side_effect=RuntimeError(
                "TorchRec backend requires the `torchrec` package to be installed."
            ),
        ), self.assertRaisesRegex(
            RuntimeError, "TorchRec backend requires the `torchrec` package"
        ):
            build_runner(cfg, Path("/tmp"))

    def test_build_runner_torchrec_returns_runner_when_dependency_available(self) -> None:
        cfg = RunConfig(backend="torchrec")
        with mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_torchrec_available",
            return_value=None,
        ):
            runner = build_runner(cfg, Path("/tmp"))
        self.assertEqual(runner.__class__.__name__, "TorchRecRunner")

    def test_runner_rejects_profiler_subargs_before_dependency_check(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            torchrec_profiler_warmup=1,
        )
        runner = TorchRecRunner(Path("/tmp"))
        with self.assertRaisesRegex(
            RuntimeError, "TorchRec profiler sub-arguments require --torchrec-profiler"
        ):
            runner.run(Path("/app/RecStore"), cfg)

    def test_runner_rejects_non_torchrec_backend(self) -> None:
        runner = TorchRecRunner(Path("/tmp"))
        with self.assertRaisesRegex(
            ValueError, "TorchRecRunner requires cfg.backend to be 'torchrec'"
        ):
            runner.run(Path("/app/RecStore"), RunConfig(backend="recstore", steps=1))

    def test_runner_builds_multi_node_torchrun_command(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=2,
            nnodes=2,
            node_rank=1,
            nproc=4,
            nproc_per_node=4,
            master_addr="10.0.2.191",
            master_port=29600,
            rdzv_backend="c10d",
            rdzv_id="demo-run",
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="case-a",
            torchrec_main_csv="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_main.csv",
            torchrec_main_agg_csv="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_main_agg.csv",
            torchrec_trace_dir="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_traces",
            torchrec_trace_csv="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_trace.csv",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)
        self.assertIn("--nnodes", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--node_rank", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--nproc_per_node", cmd)
        self.assertIn("4", cmd)
        self.assertIn("--rdzv_backend", cmd)
        self.assertIn("c10d", cmd)
        self.assertIn("--rdzv_endpoint", cmd)
        self.assertIn("10.0.2.191:29600", cmd)
        self.assertIn("--rdzv_id", cmd)
        self.assertIn("demo-run", cmd)
        self.assertIn("--tee", cmd)
        self.assertIn("3", cmd)
        self.assertNotIn("--standalone", cmd)
        self.assertIn("--master-addr", cmd)
        self.assertIn("--master-port", cmd)
        self.assertIn("--rdzv-backend", cmd)
        self.assertIn("--rdzv-id", cmd)
        self.assertIn("--output-root", cmd)
        self.assertIn("--run-id", cmd)

    def test_runner_builds_fair_remote_torchrun_command(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nnodes=2,
            node_rank=0,
            nproc=1,
            nproc_per_node=1,
            master_addr="10.0.2.196",
            master_port=29611,
            rdzv_backend="c10d",
            rdzv_id="fair-case",
            output_root="/tmp/rs_demo",
            run_id="fair-case",
            torchrec_main_csv="/tmp/rs_demo/out.csv",
            torchrec_main_agg_csv="/tmp/rs_demo/out_agg.csv",
            torchrec_trace_dir="/tmp/rs_demo/traces",
            torchrec_trace_csv="/tmp/rs_demo/trace.csv",
            torchrec_dist_mode="fair_remote",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)
        self.assertIn("--torchrec-dist-mode", cmd)
        self.assertIn("fair_remote", cmd)

    def test_runner_forwards_torchrec_memory_mode_to_worker(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nnodes=2,
            node_rank=0,
            nproc=1,
            nproc_per_node=1,
            master_addr="10.0.2.196",
            master_port=29611,
            rdzv_backend="c10d",
            rdzv_id="uvm-case",
            output_root="/tmp/rs_demo",
            run_id="uvm-case",
            torchrec_main_csv="/tmp/rs_demo/out.csv",
            torchrec_main_agg_csv="/tmp/rs_demo/out_agg.csv",
            torchrec_trace_dir="/tmp/rs_demo/traces",
            torchrec_trace_csv="/tmp/rs_demo/trace.csv",
            torchrec_memory_mode="uvm_caching",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)
        self.assertIn("--torchrec-memory-mode", cmd)
        self.assertIn("uvm_caching", cmd)

    def test_runner_forwards_torchrec_timing_sync_mode_to_worker(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nnodes=2,
            node_rank=0,
            nproc=1,
            nproc_per_node=1,
            master_addr="10.0.2.196",
            master_port=29611,
            rdzv_backend="c10d",
            rdzv_id="timing-case",
            output_root="/tmp/rs_demo",
            run_id="timing-case",
            torchrec_main_csv="/tmp/rs_demo/out.csv",
            torchrec_main_agg_csv="/tmp/rs_demo/out_agg.csv",
            torchrec_trace_dir="/tmp/rs_demo/traces",
            torchrec_trace_csv="/tmp/rs_demo/trace.csv",
            torchrec_timing_sync_mode="step",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)
        self.assertIn("--torchrec-timing-sync-mode", cmd)
        self.assertIn("step", cmd)

    def test_build_uvm_caching_constraints_uses_all_table_names(self) -> None:
        constraints = _build_uvm_caching_constraints(
            table_names=["t_cat_0", "t_cat_1"],
            parameter_constraints_cls=_FakeParameterConstraints,
            embedding_compute_kernel_cls=_FakeEmbeddingComputeKernel,
        )

        self.assertEqual(set(constraints), {"t_cat_0", "t_cat_1"})
        self.assertEqual(
            constraints["t_cat_0"].compute_kernels,
            ["fused_uvm_caching"],
        )

    def test_runner_forwards_dense_arch_args_to_worker(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nnodes=2,
            node_rank=0,
            nproc=1,
            nproc_per_node=1,
            master_addr="10.0.2.196",
            master_port=29611,
            rdzv_backend="c10d",
            rdzv_id="arch-case",
            output_root="/tmp/rs_demo",
            run_id="arch-case",
            torchrec_main_csv="/tmp/rs_demo/out.csv",
            torchrec_main_agg_csv="/tmp/rs_demo/out_agg.csv",
            torchrec_trace_dir="/tmp/rs_demo/traces",
            torchrec_trace_csv="/tmp/rs_demo/trace.csv",
            embedding_dim=16,
            dense_arch_layer_sizes="64,32,16",
            over_arch_layer_sizes="128,64,1",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)
        self.assertIn("--dense-arch-layer-sizes", cmd)
        self.assertIn("64,32,16", cmd)
        self.assertIn("--over-arch-layer-sizes", cmd)
        self.assertIn("128,64,1", cmd)

    def test_runner_rank_output_dir_uses_shared_output_root(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nnodes=2,
            nproc_per_node=2,
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="case-b",
            torchrec_main_csv="/nas/home/shq/docker/rs_demo/outputs/case-b/torchrec_main.csv",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        rank_dir = runner._rank_output_dir(cfg)
        self.assertEqual(
            rank_dir,
            Path("/nas/home/shq/docker/rs_demo/outputs/case-b/torchrec_ranks"),
        )

    def test_runner_uses_world_size_from_nnodes_and_nproc_per_node(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nproc=1,
            nnodes=2,
            nproc_per_node=2,
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="case-c",
            torchrec_main_csv="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_main.csv",
            torchrec_main_agg_csv="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_main_agg.csv",
            torchrec_trace_dir="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_traces",
            torchrec_trace_csv="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_trace.csv",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        with mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_torchrec_available",
            return_value=None,
        ), mock.patch.object(runner, "_run_distributed", return_value={"backend": "torchrec", "rows": []}) as dist_run:
            result = runner.run(Path("/app/RecStore"), cfg)
        self.assertEqual(result["backend"], "torchrec")
        dist_run.assert_called_once()

    def test_runner_uses_torchrun_for_single_process_uvm_caching(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nproc=1,
            nnodes=1,
            nproc_per_node=1,
            torchrec_memory_mode="uvm_caching",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        with mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_torchrec_available",
            return_value=None,
        ), mock.patch.object(
            runner, "_run_distributed", return_value={"backend": "torchrec", "rows": []}
        ) as dist_run, mock.patch.object(
            runner, "_run_single_process", return_value={"backend": "torchrec", "rows": []}
        ) as single_run:
            result = runner.run(Path("/app/RecStore"), cfg)

        self.assertEqual(result["backend"], "torchrec")
        dist_run.assert_called_once()
        single_run.assert_not_called()

    def test_parse_nccl_transport_log(self) -> None:
        cases = [
            (
                "node:1:2 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [RO]; "
                "OOB enp3s0f0:10.0.2.192<0>\n",
                "RDMA",
            ),
            (
                "node:1:2 [0] NCCL INFO NET/Socket : Using [0]enp3s0f0:10.0.2.192<0>\n",
                "TCP",
            ),
        ]
        for sample, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", delete=False
                ) as handle:
                    handle.write(sample)
                    path = Path(handle.name)
                try:
                    self.assertEqual(_parse_nccl_transport_log(path), expected)
                finally:
                    path.unlink()

    def test_runner_sets_explicit_socket_env_for_multi_node(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nnodes=2,
            node_rank=0,
            nproc_per_node=1,
            master_addr="10.0.2.196",
            master_port=29611,
            rdzv_backend="c10d",
            rdzv_id="env-case",
            output_root="/tmp/rs_demo",
            run_id="env-case",
            torchrec_main_csv="/tmp/rs_demo/out.csv",
            torchrec_main_agg_csv="/tmp/rs_demo/out_agg.csv",
            torchrec_trace_dir="/tmp/rs_demo/traces",
            torchrec_trace_csv="/tmp/rs_demo/trace.csv",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        fake_result = mock.Mock(returncode=0, stdout="", stderr="")
        run_mock = mock.Mock(return_value=fake_result)
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"}
        }
        with mock.patch.dict(os.environ, clean_env, clear=True), mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_torchrec_available",
            return_value=None,
        ), mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner._pick_socket_ifname",
            return_value="eno1",
        ), mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.subprocess.run",
            run_mock,
        ), mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner._merge_rank_outputs",
            return_value=[],
        ), mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_shared_dir",
            return_value=None,
        ), mock.patch(
            "pathlib.Path.exists",
            return_value=True,
        ):
            runner.run(Path("/app/RecStore"), cfg)

        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "eno1")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "eno1")

    def test_distributed_run_removes_stale_coordination_files_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            run_id = "stale-case"
            run_output = output_root / "outputs" / run_id
            rank_dir = run_output / "torchrec_ranks"
            rank_dir.mkdir(parents=True)
            stale_paths = [
                run_output / "torchrec_worker_fingerprints.json",
                run_output / "torchrec_worker_fingerprints.json.lock",
                run_output / "torchrec_plan.pkl",
                rank_dir / "rank0.csv",
            ]
            for path in stale_paths:
                path.write_text("stale", encoding="utf-8")

            cfg = RunConfig(
                backend="torchrec",
                steps=1,
                nnodes=1,
                node_rank=0,
                nproc_per_node=1,
                master_addr="127.0.0.1",
                master_port=29612,
                rdzv_id=run_id,
                output_root=str(output_root),
                run_id=run_id,
                torchrec_main_csv=str(run_output / "torchrec_main.csv"),
                torchrec_main_agg_csv=str(run_output / "torchrec_main_agg.csv"),
                torchrec_trace_dir=str(run_output / "torchrec_traces"),
                torchrec_trace_csv=str(run_output / "torchrec_trace.csv"),
            )
            runner = TorchRecRunner(Path("/tmp/runtime"))

            def fake_subprocess_run(*args, **kwargs):
                for path in stale_paths:
                    self.assertFalse(path.exists(), f"{path} was not cleaned")
                write_stage_csv(
                    rank_dir / "rank0.csv",
                    [
                        {
                            "backend": "torchrec",
                            "rank": 0,
                            "step": 0,
                            "warmup_excluded": 0,
                        }
                    ],
                )
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch(
                "model_zoo.rs_demo.runners.torchrec_runner._pick_socket_ifname",
                return_value=None,
            ), mock.patch(
                "model_zoo.rs_demo.runners.torchrec_runner.subprocess.run",
                side_effect=fake_subprocess_run,
            ), mock.patch(
                "model_zoo.rs_demo.runners.torchrec_runner._merge_rank_outputs",
                return_value=[],
            ):
                result = runner._run_distributed(Path("/app/RecStore"), cfg)

        self.assertEqual(result["backend"], "torchrec")

    def test_merge_rank_outputs_keeps_only_trainer_rows_for_fair_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path0 = Path(tmpdir) / "rank0.csv"
            path1 = Path(tmpdir) / "rank1.csv"
            out_path = Path(tmpdir) / "merged.csv"
            write_stage_csv(
                path0,
                [
                    {
                        "backend": "torchrec",
                        "rank": 0,
                        "step": 0,
                        "collective_mode": "measured_distributed",
                        "collective_measured": 1,
                        "torchrec_dist_mode": "fair_remote",
                        "torchrec_role": "trainer",
                        "torchrec_is_trainer": 1,
                        "step_total_ms": 10.0,
                    }
                ],
            )
            write_stage_csv(
                path1,
                [
                    {
                        "backend": "torchrec",
                        "rank": 1,
                        "step": 0,
                        "collective_mode": "measured_distributed",
                        "collective_measured": 1,
                        "torchrec_dist_mode": "fair_remote",
                        "torchrec_role": "embedding_worker",
                        "torchrec_is_trainer": 0,
                        "step_total_ms": 20.0,
                    }
                ],
            )

            rows = _merge_rank_outputs([path0, path1], out_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rank"], 0)
        self.assertEqual(rows[0]["torchrec_role"], "trainer")

    def test_build_train_dataloader_for_fair_remote_disables_shuffle(self) -> None:
        fake_dataset = [1, 2, 3]

        with mock.patch(
            "model_zoo.rs_demo.data.dlrm_source.build_train_dataloader",
            return_value=(fake_dataset, "loader"),
        ) as build_loader:
            dataset, dataloader = _build_train_dataloader_for_mode(
                repo_root=Path("/app/RecStore"),
                cfg=RunConfig(
                    backend="torchrec",
                    steps=1,
                    nnodes=2,
                    nproc_per_node=1,
                    torchrec_dist_mode="fair_remote",
                ),
                rank=1,
                torch=torch,
            )

        self.assertEqual(dataset, fake_dataset)
        self.assertEqual(dataloader, "loader")
        self.assertEqual(build_loader.call_args.kwargs["shuffle"], False)
        self.assertEqual(build_loader.call_args.kwargs["seed"], 20260330)

    def test_build_train_dataloader_for_replicated_distributed_shards_by_rank(self) -> None:
        fake_dataset = [1, 2, 3]

        with mock.patch(
            "model_zoo.rs_demo.data.dlrm_source.build_train_dataloader",
            return_value=(fake_dataset, "loader"),
        ) as build_loader:
            dataset, dataloader = _build_train_dataloader_for_mode(
                repo_root=Path("/app/RecStore"),
                cfg=RunConfig(
                    backend="torchrec",
                    steps=1,
                    nnodes=2,
                    nproc_per_node=2,
                    torchrec_dist_mode="replicated",
                ),
                rank=3,
                torch=torch,
            )

        self.assertEqual(dataset, fake_dataset)
        self.assertEqual(dataloader, "loader")
        self.assertEqual(build_loader.call_args.kwargs["shuffle"], True)
        self.assertEqual(build_loader.call_args.kwargs["seed"], 20260330)
        self.assertEqual(build_loader.call_args.kwargs["rank"], 3)
        self.assertEqual(build_loader.call_args.kwargs["world_size"], 4)

    def test_maybe_wrap_dense_module_for_dist_wraps_replicated_dense(self) -> None:
        dense_module = mock.Mock()
        fake_ddp = mock.sentinel.ddp
        device = mock.Mock()
        device.type = "cuda"

        with mock.patch(
            "torch.nn.parallel.DistributedDataParallel",
            return_value=fake_ddp,
        ) as ddp_cls:
            wrapped = _maybe_wrap_dense_module_for_dist(
                dense_module=dense_module,
                device=device,
                local_rank=1,
                use_dist=True,
                fair_remote_mode=False,
                torch=torch,
            )

        self.assertIs(wrapped, fake_ddp)
        ddp_cls.assert_called_once_with(
            dense_module,
            device_ids=[1],
            output_device=1,
        )


if __name__ == "__main__":
    unittest.main()
