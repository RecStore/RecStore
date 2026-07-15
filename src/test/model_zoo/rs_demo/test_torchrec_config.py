from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_zoo.rs_demo import cli
from model_zoo.rs_demo.config import (
    ensure_run_id,
    parse_config,
    populate_default_paths,
    resolve_num_embeddings_per_feature,
    validate_hps_torch_config,
    validate_recstore_config,
    validate_torchrec_config,
)


class TestTorchRecConfig(unittest.TestCase):
    def test_recstore_distributed_allows_multi_node(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "recstore",
                "--nnodes",
                "2",
                "--node-rank",
                "0",
                "--nproc-per-node",
                "1",
                "--recstore-runtime-dir",
                "/tmp/recstore-shared-runtime",
            ]
        )
        validate_recstore_config(cfg)

    def test_torchrec_distributed_fields_parse(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--nnodes",
                "2",
                "--node-rank",
                "1",
                "--nproc-per-node",
                "4",
                "--master-addr",
                "10.0.2.191",
                "--master-port",
                "29600",
                "--rdzv-backend",
                "c10d",
                "--rdzv-id",
                "demo-run",
                "--output-root",
                "/nas/home/shq/docker/rs_demo",
                "--run-id",
                "case-a",
            ]
        )
        self.assertEqual(cfg.nnodes, 2)
        self.assertEqual(cfg.node_rank, 1)
        self.assertEqual(cfg.nproc_per_node, 4)
        self.assertEqual(cfg.master_addr, "10.0.2.191")
        self.assertEqual(cfg.master_port, 29600)
        self.assertEqual(cfg.rdzv_backend, "c10d")
        self.assertEqual(cfg.rdzv_id, "demo-run")
        self.assertEqual(cfg.output_root, "/nas/home/shq/docker/rs_demo")
        self.assertEqual(cfg.run_id, "case-a")

    def test_num_embeddings_per_feature_override_parses(self) -> None:
        values = [str(idx + 1) for idx in range(26)]
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--num-embeddings",
                "5000",
                "--num-embeddings-per-feature",
                ",".join(values),
            ]
        )

        self.assertEqual(resolve_num_embeddings_per_feature(
            cfg.num_embeddings,
            cfg.num_embeddings_per_feature,
        ), list(range(1, 27)))

    def test_torchrec_dist_mode_parses(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-dist-mode",
                "fair_remote",
            ]
        )
        self.assertEqual(cfg.torchrec_dist_mode, "fair_remote")

    def test_torchrec_memory_mode_parses_uvm_caching(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-memory-mode",
                "uvm_caching",
            ]
        )
        self.assertEqual(cfg.torchrec_memory_mode, "uvm_caching")

    def test_torchrec_timing_sync_mode_parses_step(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-timing-sync-mode",
                "step",
            ]
        )
        self.assertEqual(cfg.torchrec_timing_sync_mode, "step")

    def test_torchrec_align_recstore_init_parses(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-align-recstore-init",
            ]
        )

        self.assertTrue(cfg.torchrec_align_recstore_init)

    def test_recstore_ps_type_accepts_rdma(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "recstore",
                "--ps-type",
                "RDMA",
            ]
        )

        self.assertEqual(cfg.ps_type, "RDMA")
        validate_recstore_config(cfg)

    def test_hps_torch_backend_parses_paths(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "hps_torch",
                "--hps-torch-model-name",
                "dlrm_hps",
                "--hps-torch-config-file",
                "/tmp/hps.json",
                "--hps-torch-model-dir",
                "/tmp/hps_model",
                "--hps-torch-main-csv",
                "/tmp/hps.csv",
                "--hps-torch-main-agg-csv",
                "/tmp/hps_agg.csv",
                "--hps-torch-gpucacheper",
                "0.5",
            ]
        )
        self.assertEqual(cfg.backend, "hps_torch")
        self.assertEqual(cfg.hps_torch_model_name, "dlrm_hps")
        self.assertEqual(cfg.hps_torch_config_file, "/tmp/hps.json")
        self.assertEqual(cfg.hps_torch_model_dir, "/tmp/hps_model")
        self.assertEqual(cfg.hps_torch_main_csv, "/tmp/hps.csv")
        self.assertEqual(cfg.hps_torch_main_agg_csv, "/tmp/hps_agg.csv")
        self.assertEqual(cfg.hps_torch_gpucacheper, 0.5)

    def test_hps_torch_backend_accepts_single_node_multi_process(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "hps_torch",
                "--nproc-per-node",
                "2",
            ]
        )
        validate_hps_torch_config(cfg)

    def test_hps_torch_backend_rejects_multi_node(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "hps_torch",
                "--nnodes",
                "2",
                "--node-rank",
                "0",
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "single-node"):
            validate_hps_torch_config(cfg)

    def test_torchrec_backend_parses_profiler_flags(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-profiler",
                "--torchrec-profiler-warmup",
                "1",
                "--torchrec-profiler-active",
                "3",
                "--torchrec-profiler-repeat",
                "2",
                "--torchrec-trace-dir",
                "/tmp/example/trace",
                "--torchrec-main-csv",
                "/tmp/example/main.csv",
                "--torchrec-main-agg-csv",
                "/tmp/example/main_agg.csv",
                "--torchrec-trace-csv",
                "/tmp/example/trace.csv",
            ]
        )
        self.assertEqual(cfg.backend, "torchrec")
        self.assertEqual(cfg.nproc, 1)
        self.assertTrue(cfg.torchrec_profiler)
        self.assertEqual(cfg.torchrec_profiler_warmup, 1)
        self.assertEqual(cfg.torchrec_profiler_active, 3)
        self.assertEqual(cfg.torchrec_profiler_repeat, 2)
        self.assertEqual(cfg.torchrec_trace_dir, "/tmp/example/trace")
        self.assertEqual(cfg.torchrec_main_csv, "/tmp/example/main.csv")
        self.assertEqual(cfg.torchrec_main_agg_csv, "/tmp/example/main_agg.csv")
        self.assertEqual(cfg.torchrec_trace_csv, "/tmp/example/trace.csv")

    def test_torchrec_no_start_server_flag(self) -> None:
        cfg = parse_config(["--backend", "torchrec", "--nproc", "4", "--no-start-server"])
        self.assertEqual(cfg.backend, "torchrec")
        self.assertEqual(cfg.nproc, 4)
        self.assertFalse(cfg.start_server)

    def test_parse_config_defaults_nproc_per_node_to_nproc(self) -> None:
        cfg = parse_config(["--backend", "torchrec", "--nproc", "4"])
        self.assertEqual(cfg.nproc_per_node, 4)

    def test_torchrec_nproc_per_node_must_be_positive(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--nproc-per-node must be greater than 0"):
            validate_torchrec_config(
                parse_config(["--backend", "torchrec", "--nproc-per-node", "0"])
            )

    def test_torchrec_node_rank_must_be_in_range(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--node-rank must be within \\[0, nnodes\\)"):
            validate_torchrec_config(
                parse_config(
                    ["--backend", "torchrec", "--nnodes", "2", "--node-rank", "2"]
                )
            )

    def test_torchrec_profiler_allows_subargs_when_enabled(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-profiler",
                "--torchrec-profiler-warmup",
                "1",
                "--torchrec-profiler-active",
                "3",
                "--torchrec-profiler-repeat",
                "2",
                "--torchrec-trace-dir",
                "/tmp/example/trace",
                "--torchrec-trace-csv",
                "/tmp/example/trace.csv",
            ]
        )
        validate_torchrec_config(cfg)

    def test_torchrec_profiler_subargs_require_profiler_flag(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "TorchRec profiler sub-arguments require --torchrec-profiler"
        ):
            cfg = parse_config(
                [
                    "--backend",
                    "torchrec",
                    "--torchrec-profiler-warmup",
                    "1",
                ]
            )
            validate_torchrec_config(cfg)

    def test_torchrec_fair_remote_requires_multi_process_world(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-dist-mode",
                "fair_remote",
                "--nnodes",
                "1",
                "--nproc-per-node",
                "1",
            ]
        )
        with self.assertRaisesRegex(
            RuntimeError, "fair_remote requires world_size greater than 1"
        ):
            validate_torchrec_config(cfg)

    def test_ensure_run_id_generates_when_missing(self) -> None:
        cfg = parse_config(["--backend", "torchrec"])
        cfg.run_id = ""
        ensure_run_id(cfg)
        self.assertTrue(cfg.run_id.startswith("run_"))

    def test_default_output_root_uses_shared_nas_path(self) -> None:
        cfg = parse_config(["--backend", "torchrec"])
        self.assertEqual(cfg.output_root, "/nas/home/shq/docker/rs_demo")

    def test_populate_default_paths_moves_outputs_to_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = parse_config(["--backend", "torchrec"])
            cfg.output_root = tmpdir
            cfg.run_id = "run_test"
            populate_default_paths(cfg)
            self.assertEqual(
                cfg.jsonl, str(Path(tmpdir) / "outputs" / "run_test" / "recstore_events.jsonl")
            )
            self.assertEqual(
                cfg.csv, str(Path(tmpdir) / "outputs" / "run_test" / "recstore_embupdate.csv")
            )
            self.assertEqual(
                cfg.torchrec_main_csv,
                str(Path(tmpdir) / "outputs" / "run_test" / "torchrec_main.csv"),
            )
            self.assertEqual(
                cfg.torchrec_main_agg_csv,
                str(Path(tmpdir) / "outputs" / "run_test" / "torchrec_main_agg.csv"),
            )
            self.assertEqual(
                cfg.torchrec_trace_dir,
                str(Path(tmpdir) / "outputs" / "run_test" / "torchrec_traces"),
            )
            self.assertEqual(
                cfg.torchrec_trace_csv,
                str(Path(tmpdir) / "outputs" / "run_test" / "torchrec_trace.csv"),
            )
            self.assertEqual(
                cfg.server_log, str(Path(tmpdir) / "logs" / "run_test" / "ps_server.log")
            )

    def test_populate_default_paths_respects_explicit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = parse_config(["--backend", "torchrec", "--jsonl", "/tmp/custom.jsonl"])
            cfg.output_root = tmpdir
            cfg.run_id = "run_test"
            populate_default_paths(cfg)
            self.assertEqual(cfg.jsonl, "/tmp/custom.jsonl")

    def test_cli_writes_trace_csv_only_when_profiler_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_root = Path(tmpdir) / "traces"
            trace_csv = Path(tmpdir) / "trace.csv"
            main_csv = Path(tmpdir) / "main.csv"
            main_agg_csv = Path(tmpdir) / "main_agg.csv"

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    main_rows = [
                        {
                            "backend": "torchrec",
                            "batch_size": 2,
                            "step": 0,
                            "warmup_excluded": 0,
                            "collective_mode": "not_measured_single_process",
                            "collective_measured": 0,
                            "step_total_ms": 10.0,
                            "batch_prepare_ms": 1.0,
                            "input_pack_ms": 0.5,
                            "embed_lookup_local_ms": 2.0,
                            "embed_pool_local_ms": 1.0,
                            "collective_launch_ms": 0.0,
                            "collective_wait_ms": 0.0,
                            "output_unpack_ms": 0.5,
                            "dense_fwd_ms": 1.0,
                            "backward_ms": 2.0,
                            "optimizer_ms": 1.0,
                            "collective_total_ms": 0.0,
                            "network_proxy_torchrec_ms": 0.0,
                            "kv_local_only_ms": 3.0,
                            "kv_extended_ms": 4.0,
                            "network_proxy_torchrec_extended_ms": 1.0,
                        }
                    ]
                    with Path(cfg.torchrec_main_csv).open("w", encoding="utf-8") as f:
                        f.write(",".join(main_rows[0].keys()) + "\n")
                        f.write(",".join(str(v) for v in main_rows[0].values()) + "\n")
                    trace_dir = Path(cfg.torchrec_trace_dir)
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    (trace_dir / "sample.pt.trace.json").write_text(
                        json.dumps(
                            {"traceEvents": [{"name": "cudaStreamSynchronize", "dur": 1000}]}
                        ),
                        encoding="utf-8",
                    )
                    return {"backend": "torchrec", "rows": []}

            with mock.patch.object(cli, "build_runner", return_value=_FakeRunner()):
                rc = cli.main(
                    [
                        "--backend",
                        "torchrec",
                        "--steps",
                        "1",
                        "--no-start-server",
                        "--torchrec-profiler",
                        "--torchrec-trace-dir",
                        str(trace_root),
                        "--torchrec-main-csv",
                        str(main_csv),
                        "--torchrec-main-agg-csv",
                        str(main_agg_csv),
                        "--torchrec-trace-csv",
                        str(trace_csv),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertTrue(trace_csv.exists())
            self.assertTrue(main_agg_csv.exists())

    def test_cli_does_not_write_trace_csv_when_profiler_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "no-profiler"
            output_root = Path(tmpdir)

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    with Path(cfg.torchrec_main_csv).open("w", encoding="utf-8") as f:
                        f.write("step_total_ms,collective_launch_ms,collective_wait_ms,collective_total_ms,kv_local_only_ms,kv_extended_ms,input_pack_ms,output_unpack_ms\n")
                        f.write("1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n")
                    return {"backend": "torchrec", "rows": []}

            with mock.patch.object(cli, "build_runner", return_value=_FakeRunner()):
                rc = cli.main(
                    [
                        "--backend",
                        "torchrec",
                        "--steps",
                        "1",
                        "--no-start-server",
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        run_id,
                    ]
                )

            self.assertEqual(rc, 0)
            default_trace_csv = output_root / "outputs" / run_id / "torchrec_trace.csv"
            default_main_agg_csv = output_root / "outputs" / run_id / "torchrec_main_agg.csv"
            self.assertFalse(default_trace_csv.exists())
            self.assertTrue(default_main_agg_csv.exists())

    def test_cli_recstore_worker_skips_post_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {
                "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
                "cache_ps": {"servers": []},
                "distributed_client": {"servers": []},
            }
            (Path(tmpdir) / "recstore_config.json").write_text(
                json.dumps(base_cfg),
                encoding="utf-8",
            )

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    return {"backend": "recstore", "rows": []}

            with mock.patch.dict("os.environ", {"RS_DEMO_RECSTORE_WORKER": "1"}, clear=False):
                with mock.patch.object(cli, "build_runner", return_value=_FakeRunner()):
                    with mock.patch.object(cli, "repo_root_from_this_file", return_value=Path(tmpdir)):
                        rc = cli.main(
                            [
                                "--backend",
                                "recstore",
                                "--steps",
                                "1",
                                "--no-start-server",
                                "--output-root",
                                tmpdir,
                                "--run-id",
                                "recstore-worker",
                            ]
                        )

            self.assertEqual(rc, 0)

    def test_cli_loads_base_config_from_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resolver_config = Path(tmpdir) / "recstore_config.json"
            resolver_config.write_text(
                json.dumps(
                    {
                        "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
                        "cache_ps": {"servers": []},
                        "distributed_client": {"servers": []},
                    }
                ),
                encoding="utf-8",
            )

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    Path(cfg.recstore_main_csv).parent.mkdir(parents=True, exist_ok=True)
                    Path(cfg.recstore_main_csv).write_text(
                        "step_total_ms,input_pack_ms,embed_lookup_local_ms,embed_pool_local_ms,output_unpack_ms,dense_fwd_ms,backward_ms,optimizer_ms,sparse_update_ms,emb_stage_ms\n"
                        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0\n",
                        encoding="utf-8",
                    )
                    return {"backend": "recstore", "rows": []}

            missing_repo_root = Path(tmpdir) / "missing-repo-root"
            with mock.patch.object(
                cli, "resolve_recstore_config_path", return_value=resolver_config
            ), mock.patch.object(
                cli, "repo_root_from_this_file", return_value=missing_repo_root
            ), mock.patch.object(
                cli, "build_runner", return_value=_FakeRunner()
            ), mock.patch.object(
                cli, "make_runtime_dir", return_value=(Path(tmpdir), resolver_config)
            ), mock.patch.object(
                cli, "analyze_embupdate", return_value="ok"
            ):
                rc = cli.main(
                    [
                        "--backend",
                        "recstore",
                        "--steps",
                        "1",
                        "--no-start-server",
                        "--output-root",
                        tmpdir,
                        "--run-id",
                        "resolver-config",
                    ]
                )

            self.assertEqual(rc, 0)

    def test_cli_recstore_runtime_dir_skips_make_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            base_cfg = {
                "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
                "cache_ps": {"servers": []},
                "distributed_client": {"servers": []},
            }
            (repo_root / "recstore_config.json").write_text(json.dumps(base_cfg), encoding="utf-8")
            shared_runtime = repo_root / "shared-runtime"
            shared_runtime.mkdir()
            (shared_runtime / "recstore_config.json").write_text(
                json.dumps(base_cfg),
                encoding="utf-8",
            )

            class _FakeRunner:
                def __init__(self):
                    self.runtime_dir = None

                def run(self, repo_root, cfg):
                    Path(cfg.recstore_main_csv).parent.mkdir(parents=True, exist_ok=True)
                    Path(cfg.recstore_main_csv).write_text(
                        "step_total_ms,input_pack_ms,embed_lookup_local_ms,embed_pool_local_ms,output_unpack_ms,dense_fwd_ms,backward_ms,optimizer_ms,sparse_update_ms,emb_stage_ms\n"
                        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0\n",
                        encoding="utf-8",
                    )
                    return {"backend": "recstore", "rows": []}

            fake_runner = _FakeRunner()

            with mock.patch.object(cli, "build_runner", return_value=fake_runner), mock.patch.object(
                cli, "repo_root_from_this_file", return_value=repo_root
            ), mock.patch.object(
                cli, "make_runtime_dir", side_effect=AssertionError("make_runtime_dir should not be called")
            ), mock.patch.object(
                cli, "analyze_embupdate", return_value="ok"
            ):
                rc = cli.main(
                    [
                        "--backend",
                        "recstore",
                        "--steps",
                        "1",
                        "--no-start-server",
                        "--output-root",
                        str(repo_root),
                        "--run-id",
                        "recstore-external-runtime",
                        "--recstore-runtime-dir",
                        str(shared_runtime),
                    ]
                )

            self.assertEqual(rc, 0)

    def test_cli_recstore_resolves_relative_runtime_and_csv_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            base_cfg = {
                "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
                "cache_ps": {"servers": []},
                "distributed_client": {"servers": []},
            }
            (repo_root / "recstore_config.json").write_text(
                json.dumps(base_cfg),
                encoding="utf-8",
            )
            shared_runtime = repo_root / "relative-runtime"
            shared_runtime.mkdir()
            (shared_runtime / "recstore_config.json").write_text(
                json.dumps(base_cfg),
                encoding="utf-8",
            )
            captured = {}

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    captured["runtime_dir"] = cfg.recstore_runtime_dir
                    captured["RECSTORE_CONFIG"] = os.environ.get("RECSTORE_CONFIG")
                    captured["recstore_main_csv"] = cfg.recstore_main_csv
                    Path(cfg.recstore_main_csv).parent.mkdir(parents=True, exist_ok=True)
                    Path(cfg.recstore_main_csv).write_text(
                        "step_total_ms,input_pack_ms,embed_lookup_local_ms,embed_pool_local_ms,output_unpack_ms,dense_fwd_ms,backward_ms,optimizer_ms,sparse_update_ms,emb_stage_ms\n"
                        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0\n",
                        encoding="utf-8",
                    )
                    return {"backend": "recstore", "rows": []}

            old_cwd = Path.cwd()
            try:
                os.chdir(repo_root)
                with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                    cli, "build_runner", return_value=_FakeRunner()
                ), mock.patch.object(
                    cli, "repo_root_from_this_file", return_value=repo_root
                ), mock.patch.object(
                    cli,
                    "make_runtime_dir",
                    side_effect=AssertionError("make_runtime_dir should not be called"),
                ), mock.patch.object(
                    cli, "analyze_embupdate", return_value="ok"
                ):
                    rc = cli.main(
                        [
                            "--backend",
                            "recstore",
                            "--steps",
                            "1",
                            "--no-start-server",
                            "--output-root",
                            str(repo_root),
                            "--run-id",
                            "recstore-relative-runtime",
                            "--recstore-runtime-dir",
                            "relative-runtime",
                            "--recstore-main-csv",
                            "relative-artifacts/recstore_main.csv",
                        ]
                    )
            finally:
                os.chdir(old_cwd)

            expected_runtime = shared_runtime.resolve()
            self.assertEqual(rc, 0)
            self.assertEqual(captured["runtime_dir"], str(expected_runtime))
            self.assertEqual(
                captured["RECSTORE_CONFIG"],
                str(expected_runtime / "recstore_config.json"),
            )
            self.assertEqual(
                captured["recstore_main_csv"],
                str((repo_root / "relative-artifacts/recstore_main.csv").resolve()),
            )

    def test_cli_recstore_assigns_generated_runtime_dir_back_to_cfg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            base_cfg = {
                "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
                "cache_ps": {"servers": []},
                "distributed_client": {"servers": []},
            }
            (repo_root / "recstore_config.json").write_text(json.dumps(base_cfg), encoding="utf-8")
            generated_runtime = repo_root / "runtime-generated"
            generated_runtime.mkdir()
            (generated_runtime / "recstore_config.json").write_text(
                json.dumps(base_cfg),
                encoding="utf-8",
            )

            captured_cfg = {}

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    captured_cfg["recstore_runtime_dir"] = cfg.recstore_runtime_dir
                    Path(cfg.recstore_main_csv).parent.mkdir(parents=True, exist_ok=True)
                    Path(cfg.recstore_main_csv).write_text(
                        "step_total_ms,input_pack_ms,embed_lookup_local_ms,embed_pool_local_ms,output_unpack_ms,dense_fwd_ms,backward_ms,optimizer_ms,sparse_update_ms,emb_stage_ms\n"
                        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0\n",
                        encoding="utf-8",
                    )
                    return {"backend": "recstore", "rows": []}

            with mock.patch.object(cli, "build_runner", return_value=_FakeRunner()), mock.patch.object(
                cli, "repo_root_from_this_file", return_value=repo_root
            ), mock.patch.object(
                cli, "make_runtime_dir", return_value=(generated_runtime, generated_runtime / "recstore_config.json")
            ), mock.patch.object(
                cli, "analyze_embupdate", return_value="ok"
            ):
                rc = cli.main(
                    [
                        "--backend",
                        "recstore",
                        "--steps",
                        "1",
                        "--no-start-server",
                        "--output-root",
                        str(repo_root),
                        "--run-id",
                        "recstore-generated-runtime",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(captured_cfg["recstore_runtime_dir"], str(generated_runtime))

    def test_cli_recstore_exports_generated_runtime_config_to_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            base_cfg = {
                "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
                "cache_ps": {"servers": []},
                "distributed_client": {"servers": []},
            }
            (repo_root / "recstore_config.json").write_text(json.dumps(base_cfg), encoding="utf-8")
            generated_runtime = repo_root / "runtime-generated"
            generated_runtime.mkdir()
            generated_config = generated_runtime / "recstore_config.json"
            generated_config.write_text(json.dumps(base_cfg), encoding="utf-8")

            captured_env = {}

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    captured_env["RECSTORE_CONFIG"] = os.environ.get("RECSTORE_CONFIG")
                    Path(cfg.recstore_main_csv).parent.mkdir(parents=True, exist_ok=True)
                    Path(cfg.recstore_main_csv).write_text(
                        "step_total_ms,input_pack_ms,embed_lookup_local_ms,embed_pool_local_ms,output_unpack_ms,dense_fwd_ms,backward_ms,optimizer_ms,sparse_update_ms,emb_stage_ms\n"
                        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0\n",
                        encoding="utf-8",
                    )
                    return {"backend": "recstore", "rows": []}

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                cli, "build_runner", return_value=_FakeRunner()
            ), mock.patch.object(
                cli, "repo_root_from_this_file", return_value=repo_root
            ), mock.patch.object(
                cli, "make_runtime_dir", return_value=(generated_runtime, generated_config)
            ), mock.patch.object(
                cli, "analyze_embupdate", return_value="ok"
            ):
                rc = cli.main(
                    [
                        "--backend",
                        "recstore",
                        "--steps",
                        "1",
                        "--no-start-server",
                        "--output-root",
                        str(repo_root),
                        "--run-id",
                        "recstore-generated-runtime",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(captured_env["RECSTORE_CONFIG"], str(generated_config))

    def test_cli_recstore_rdma_uses_petps_server_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            base_cfg = {
                "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
                "cache_ps": {"servers": []},
                "distributed_client": {"servers": []},
            }
            config_path = repo_root / "recstore_config.json"
            config_path.write_text(json.dumps(base_cfg), encoding="utf-8")
            runtime_dir = repo_root / "runtime-generated"
            runtime_dir.mkdir()
            runtime_config = runtime_dir / "recstore_config.json"
            runtime_config.write_text(json.dumps(base_cfg), encoding="utf-8")
            captured = {}

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    captured["RECSTORE_CONFIG"] = os.environ.get("RECSTORE_CONFIG")
                    captured["RECSTORE_RDMA_RC_NAMESPACE"] = os.environ.get(
                        "RECSTORE_RDMA_RC_NAMESPACE"
                    )
                    captured["RECSTORE_RDMA_CONTROL_PLANE_PORT"] = os.environ.get(
                        "RECSTORE_RDMA_CONTROL_PLANE_PORT"
                    )
                    captured["RECSTORE_RDMA_GET_RESPONSE_MODE"] = os.environ.get(
                        "RECSTORE_RDMA_GET_RESPONSE_MODE"
                    )
                    Path(cfg.recstore_main_csv).parent.mkdir(parents=True, exist_ok=True)
                    Path(cfg.recstore_main_csv).write_text(
                        "step_total_ms,input_pack_ms,embed_lookup_local_ms,embed_pool_local_ms,output_unpack_ms,dense_fwd_ms,backward_ms,optimizer_ms,sparse_update_ms,emb_stage_ms\n"
                        "1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0\n",
                        encoding="utf-8",
                    )
                    return {"backend": "recstore", "rows": []}

            fake_rdma_cluster = type(
                "FakeRdmaCluster",
                (),
                {
                    "rdma_namespace": "test-rdma-ns",
                    "rdma_control_plane_host": "127.0.0.1",
                    "rdma_control_plane_port": 32123,
                },
            )()

            with mock.patch.object(
                cli, "resolve_recstore_config_path", return_value=config_path
            ), mock.patch.object(
                cli, "repo_root_from_this_file", return_value=repo_root
            ), mock.patch.object(
                cli, "make_runtime_dir", return_value=(runtime_dir, runtime_config)
            ), mock.patch.object(
                cli, "build_runner", return_value=_FakeRunner()
            ), mock.patch.object(
                cli, "start_server", side_effect=AssertionError("ps_server path must not be used")
            ), mock.patch.object(
                cli, "start_rdma_server_cluster", return_value=fake_rdma_cluster
            ) as start_rdma, mock.patch.object(
                cli, "stop_server", side_effect=AssertionError("ps_server stop must not be used")
            ), mock.patch.object(
                cli, "stop_rdma_server_cluster"
            ) as stop_rdma, mock.patch.object(
                cli, "analyze_embupdate", return_value="ok"
            ):
                rc = cli.main(
                    [
                        "--backend",
                        "recstore",
                        "--ps-type",
                        "RDMA",
                        "--steps",
                        "1",
                        "--output-root",
                        str(repo_root),
                        "--run-id",
                        "recstore-rdma",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(captured["RECSTORE_CONFIG"], str(runtime_config))
            self.assertEqual(captured["RECSTORE_RDMA_RC_NAMESPACE"], "test-rdma-ns")
            self.assertEqual(captured["RECSTORE_RDMA_CONTROL_PLANE_PORT"], "32123")
            self.assertEqual(
                captured["RECSTORE_RDMA_GET_RESPONSE_MODE"], "direct_sg"
            )
            self.assertEqual(start_rdma.call_count, 1)
            self.assertEqual(stop_rdma.call_count, 1)


if __name__ == "__main__":
    unittest.main()
