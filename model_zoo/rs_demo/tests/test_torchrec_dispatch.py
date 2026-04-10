from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest import mock
import unittest

from model_zoo.rs_demo.cli import build_runner
from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners.torchrec_runner import (
    TorchRecRunner,
    _barrier_for_step_alignment,
    _build_worker_fingerprint,
    _debug_log_path,
    _write_or_verify_worker_fingerprint,
    _compute_or_load_shared_sharding_plan,
    _summarize_sharding_plan,
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
                "model_zoo/rs_demo/runners/torchrec_runner.py",
                "model_zoo/rs_demo/runtime/hybrid_dlrm.py",
            ):
                path = repo_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rel_path, encoding="utf-8")

            fingerprint = _build_worker_fingerprint(repo_root)

        self.assertIn("files", fingerprint)
        self.assertIn("model_zoo/rs_demo/config.py", fingerprint["files"])
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
        with mock.patch.object(runner, "_run_distributed", return_value={"backend": "torchrec", "rows": []}) as dist_run:
            result = runner.run(Path("/app/RecStore"), cfg)
        self.assertEqual(result["backend"], "torchrec")
        dist_run.assert_called_once()

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
        with mock.patch(
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
        self.assertEqual(env["NCCL_IB_DISABLE"], "1")
        self.assertEqual(env["NCCL_SOCKET_FAMILY"], "AF_INET")


if __name__ == "__main__":
    unittest.main()
