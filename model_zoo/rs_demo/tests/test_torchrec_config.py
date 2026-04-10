from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_zoo.rs_demo import cli
from model_zoo.rs_demo.config import (
    ensure_run_id,
    parse_config,
    populate_default_paths,
    validate_recstore_config,
    validate_torchrec_config,
)


class TestTorchRecConfig(unittest.TestCase):
    def test_recstore_distributed_rejects_multi_node(self) -> None:
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
            ]
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "RecStore multi-trainer currently supports only --nnodes=1",
        ):
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


if __name__ == "__main__":
    unittest.main()
