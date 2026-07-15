from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path


class TestBenchE2E(unittest.TestCase):
    def test_build_default_plan_contains_single_gpu_and_single_node_multiprocess(self) -> None:
        from tools.benchmarks.run_bench_e2e import build_plan

        plan = build_plan(profile="smoke", output_root=Path("/tmp/rs-bench"))
        lane_slugs = [lane.slug for lane in plan.lanes]

        self.assertIn("torchrec-hbm-1p", lane_slugs)
        self.assertIn("recstore-brpc-pet-1p", lane_slugs)
        self.assertIn("recstore-rdma-pet-1p", lane_slugs)
        self.assertIn("recstore-local-shm-pet-2p", lane_slugs)
        self.assertIn("torchrec-hbm-2p", lane_slugs)

    def test_build_plan_supports_overrides_and_ablation_lanes(self) -> None:
        from tools.benchmarks.run_bench_e2e import PlanOverrides, build_plan

        plan = build_plan(
            profile="smoke",
            output_root=Path("/tmp/rs-bench"),
            overrides=PlanOverrides(
                data_rows=(8192,),
                batch_sizes=(512,),
                num_embeddings=(12345,),
                embedding_dims=(64,),
                steps=7,
                warmup_steps=2,
                repeat=2,
                include_ablation_lanes=True,
                only_lanes=("recstore-brpc-map-1p", "recstore-local-shm-pet-1p"),
            ),
        )

        self.assertEqual(plan.data_rows, (8192,))
        self.assertEqual(plan.batch_sizes, (512,))
        self.assertEqual(plan.num_embeddings, (12345,))
        self.assertEqual(plan.embedding_dims, (64,))
        self.assertEqual(plan.steps, 7)
        self.assertEqual(plan.warmup_steps, 2)
        self.assertEqual(plan.repeat, 2)
        self.assertEqual(
            [lane.slug for lane in plan.lanes],
            ["recstore-brpc-map-1p", "recstore-local-shm-pet-1p"],
        )
        local_shm_lane = plan.lanes[1]
        self.assertEqual(local_shm_lane.ps_type, "LOCAL_SHM")
        self.assertFalse(local_shm_lane.enable_single_node_fast_path)

    def test_build_plan_includes_rdma_backend_ablation_lanes(self) -> None:
        from tools.benchmarks.run_bench_e2e import PlanOverrides, build_plan

        plan = build_plan(
            profile="smoke",
            output_root=Path("/tmp/rs-bench"),
            overrides=PlanOverrides(
                include_ablation_lanes=True,
                only_lanes=(
                    "recstore-rdma-eh-1p",
                    "recstore-rdma-map-1p",
                    "recstore-rdma-pet-prefetch4-1p",
                ),
            ),
        )

        self.assertEqual(
            [lane.slug for lane in plan.lanes],
            [
                "recstore-rdma-eh-1p",
                "recstore-rdma-map-1p",
                "recstore-rdma-pet-prefetch4-1p",
            ],
        )
        self.assertTrue(all(lane.ps_type == "RDMA" for lane in plan.lanes))
        self.assertEqual(plan.lanes[0].recstore_index_type, "DRAM_EXTENDIBLE_HASH")
        self.assertEqual(plan.lanes[1].recstore_index_type, "DRAM_UNORDERED_MAP")
        self.assertEqual(plan.lanes[2].prefetch_depth, 4)

    def test_recstore_command_includes_backend_parameters(self) -> None:
        from tools.benchmarks.run_bench_e2e import ExecutionContext, E2ELane, build_rs_demo_command

        lane = E2ELane(
            slug="recstore-brpc-pet-1p",
            label="RecStore BRPC PET",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        )

        cmd = build_rs_demo_command(
            lane=lane,
            context=ExecutionContext(),
            run_id="run-x",
            data_dir=Path("/data/slice_4096"),
            output_root=Path("/tmp/out"),
            rows=4096,
            batch_size=256,
            steps=3,
            warmup_steps=1,
            num_embeddings=10000,
            embedding_dim=128,
            master_port=29600,
        )

        self.assertIn("--ps-type", cmd)
        self.assertIn("BRPC", cmd)
        self.assertIn("--recstore-index-type", cmd)
        self.assertIn("DRAM_PET_HASH", cmd)
        self.assertIn("--ps-kv-backend", cmd)
        self.assertIn("recstore_dram", cmd)

    def test_recstore_rdma_command_uses_rdma_ps_type(self) -> None:
        from tools.benchmarks.run_bench_e2e import ExecutionContext, E2ELane, build_rs_demo_command

        lane = E2ELane(
            slug="recstore-rdma-pet-1p",
            label="RecStore-RDMA-PET-1proc",
            backend="recstore",
            ps_type="RDMA",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        )

        cmd = build_rs_demo_command(
            lane=lane,
            context=ExecutionContext(),
            run_id="run-rdma",
            data_dir=Path("/data/slice_4096"),
            output_root=Path("/tmp/out"),
            rows=4096,
            batch_size=256,
            steps=3,
            warmup_steps=1,
            num_embeddings=10000,
            embedding_dim=128,
            master_port=29600,
        )

        self.assertIn("--ps-type", cmd)
        self.assertIn("RDMA", cmd)

    def test_recstore_command_supports_remote_external_ps_context(self) -> None:
        from tools.benchmarks.run_bench_e2e import (
            ExecutionContext,
            E2ELane,
            build_rs_demo_command,
            wrap_remote_command,
        )

        lane = E2ELane(
            slug="recstore-brpc-pet-1p",
            label="RecStore BRPC PET",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        )
        context = ExecutionContext(
            remote_train_host="root@10.0.2.191",
            remote_ssh_port=50201,
            remote_repo_root=Path("/remote/RecStore"),
            python_bin="/usr/bin/python3",
            nnodes=2,
            node_rank=1,
            master_addr="10.0.2.191",
            external_recstore_runtime_dir=Path("/tmp/rs-runtime"),
            no_start_recstore_server=True,
            server_host="10.0.2.190",
            server_port0=15000,
        )

        cmd = build_rs_demo_command(
            lane=lane,
            context=context,
            run_id="remote-run",
            data_dir=Path("/data/slice_4096"),
            output_root=Path("/tmp/out"),
            rows=4096,
            batch_size=256,
            steps=3,
            warmup_steps=1,
            num_embeddings=10000,
            embedding_dim=128,
            master_port=29600,
        )
        remote = wrap_remote_command(
            cmd,
            context.remote_train_host,
            cwd=context.remote_repo_root,
            ssh_port=context.remote_ssh_port,
        )

        self.assertEqual(cmd[0], "/usr/bin/python3")
        self.assertIn("/remote/RecStore/model_zoo/rs_demo/run_mock_stress.py", cmd)
        self.assertIn("--recstore-runtime-dir", cmd)
        self.assertIn("/tmp/rs-runtime", cmd)
        self.assertIn("--no-start-server", cmd)
        self.assertIn("--server-host", cmd)
        self.assertIn("10.0.2.190", cmd)
        self.assertIn("--server-port0", cmd)
        self.assertIn("15000", cmd)
        self.assertIn("--nnodes", cmd)
        self.assertIn("2", cmd)
        self.assertEqual(remote[0], "ssh")
        self.assertEqual(remote[1], "-p")
        self.assertEqual(remote[2], "50201")
        self.assertEqual(remote[3], "root@10.0.2.191")
        self.assertIn("cd /remote/RecStore &&", remote[4])

    def test_collect_e2e_summary_computes_rows_per_second(self) -> None:
        from tools.benchmarks.run_bench_e2e import collect_e2e_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "outputs" / "r1"
            run_dir.mkdir(parents=True)
            csv_path = run_dir / "torchrec_main.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "warmup_excluded",
                        "step_total_ms",
                        "embed_lookup_local_ms",
                        "sparse_update_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step": 0,
                        "warmup_excluded": 1,
                        "step_total_ms": 100,
                        "embed_lookup_local_ms": 10,
                        "sparse_update_ms": 20,
                    }
                )
                writer.writerow(
                    {
                        "step": 1,
                        "warmup_excluded": 0,
                        "step_total_ms": 50,
                        "embed_lookup_local_ms": 5,
                        "sparse_update_ms": 10,
                    }
                )

            rows = collect_e2e_summary(
                manifest=[
                    {
                        "run_id": "r1",
                        "backend": "torchrec",
                        "label": "TorchRec",
                        "rows": 4096,
                        "batch_size": 256,
                        "nproc_per_node": 1,
                        "status": "ok",
                        "main_csv": str(csv_path),
                    }
                ],
                output_root=root,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mean_step_total_ms"], 50.0)
        self.assertAlmostEqual(rows[0]["samples_per_sec"], 5120.0)
        self.assertAlmostEqual(rows[0]["lookup_mrows_per_sec"], 1.3312)

    def test_build_gap_summary_compares_recstore_best_with_torchrec(self) -> None:
        from tools.benchmarks.run_bench_e2e import build_gap_summary

        rows = [
            {
                "status": "ok",
                "label": "TorchRec-HBM-1proc",
                "backend": "torchrec",
                "torchrec_memory_mode": "hbm",
                "rows": 4096,
                "batch_size": 512,
                "num_embeddings": 50000,
                "embedding_dim": 128,
                "samples_per_sec": 1000.0,
            },
            {
                "status": "ok",
                "label": "TorchRec-UVMCache-1proc",
                "backend": "torchrec",
                "torchrec_memory_mode": "uvm_caching",
                "rows": 4096,
                "batch_size": 512,
                "num_embeddings": 50000,
                "embedding_dim": 128,
                "samples_per_sec": 800.0,
            },
            {
                "status": "ok",
                "label": "RecStore-BRPC-PET-1proc",
                "backend": "recstore",
                "rows": 4096,
                "batch_size": 512,
                "num_embeddings": 50000,
                "embedding_dim": 128,
                "samples_per_sec": 2500.0,
            },
        ]

        gap_rows = build_gap_summary(rows)

        self.assertEqual(len(gap_rows), 1)
        self.assertEqual(gap_rows[0]["best_recstore_label"], "RecStore-BRPC-PET-1proc")
        self.assertEqual(gap_rows[0]["torchrec_hbm_samples_per_sec"], 1000.0)
        self.assertEqual(gap_rows[0]["recstore_vs_hbm_speedup"], 2.5)
        self.assertEqual(gap_rows[0]["recstore_vs_uvm_speedup"], 3.125)

    def test_build_gap_summary_uses_repeat_median_per_lane(self) -> None:
        from tools.benchmarks.run_bench_e2e import build_gap_summary

        def row(label: str, backend: str, samples: float, memory_mode: str = "") -> dict[str, object]:
            return {
                "status": "ok",
                "backend": backend,
                "label": label,
                "torchrec_memory_mode": memory_mode,
                "rows": "4096",
                "batch_size": "512",
                "num_embeddings": "50000",
                "embedding_dim": "64",
                "samples_per_sec": samples,
            }

        rows = [
            row("TorchRec-HBM-1proc", "torchrec", 900.0, "hbm"),
            row("TorchRec-HBM-1proc", "torchrec", 1000.0, "hbm"),
            row("TorchRec-HBM-1proc", "torchrec", 1100.0, "hbm"),
            row("TorchRec-UVMCache-1proc", "torchrec", 700.0, "uvm_caching"),
            row("TorchRec-UVMCache-1proc", "torchrec", 800.0, "uvm_caching"),
            row("TorchRec-UVMCache-1proc", "torchrec", 900.0, "uvm_caching"),
            row("RecStore-BRPC-PET-1proc", "recstore", 1000.0),
            row("RecStore-BRPC-PET-1proc", "recstore", 2000.0),
            row("RecStore-BRPC-PET-1proc", "recstore", 9000.0),
            row("RecStore-BRPC-EH-1proc", "recstore", 2100.0),
            row("RecStore-BRPC-EH-1proc", "recstore", 2200.0),
            row("RecStore-BRPC-EH-1proc", "recstore", 2300.0),
        ]

        gap_rows = build_gap_summary(rows)

        self.assertEqual(len(gap_rows), 1)
        self.assertEqual(gap_rows[0]["best_recstore_label"], "RecStore-BRPC-EH-1proc")
        self.assertEqual(gap_rows[0]["best_recstore_samples_per_sec"], 2200.0)
        self.assertEqual(gap_rows[0]["torchrec_hbm_samples_per_sec"], 1000.0)
        self.assertEqual(gap_rows[0]["torchrec_uvm_samples_per_sec"], 800.0)

    def test_build_gap_summary_skips_unpaired_recstore_only_configs(self) -> None:
        from tools.benchmarks.run_bench_e2e import build_gap_summary

        rows = [
            {
                "status": "ok",
                "label": "RecStore-BRPC-MAP-1proc",
                "backend": "recstore",
                "rows": 131072,
                "batch_size": 512,
                "num_embeddings": 50000,
                "embedding_dim": 128,
                "samples_per_sec": 2000.0,
            },
        ]

        self.assertEqual(build_gap_summary(rows), [])

    def test_render_latex_report_aggregates_failed_log_paths_by_reason(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        rows = [
            {
                "status": "failed",
                "label": "RecStore-LOCAL_SHM-PET-1proc",
                "skip_reason": "",
                "log_path": f"/tmp/run-{idx}.log",
            }
            for idx in range(3)
        ]

        report = render_latex_report(
            summary_rows=rows,
            ps_rows=[],
            gap_rows=[],
            metadata={"gpu_count": 1, "rdma_available": False},
        )

        self.assertIn("benchmark failed; see per-run log", report)
        self.assertIn("RecStore-LOCAL\\_SHM-PET-1proc & failed & 3", report)

    def test_render_latex_report_classifies_rdma_oom_failure(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "rdma.log"
            log_path.write_text(
                "Out of memory: Killed process 1186040 (petps_server)\n",
                encoding="utf-8",
            )
            report = render_latex_report(
                summary_rows=[
                    {
                        "status": "failed",
                        "label": "RecStore-RDMA-PET-1proc",
                        "ps_type": "RDMA",
                        "log_path": str(log_path),
                    }
                ],
                ps_rows=[],
                gap_rows=[],
                metadata={"gpu_count": 1, "rdma_available": True},
            )

        self.assertIn("OOM during server/client startup", report)
        self.assertIn("RDMA 失败点不插值", report)

    def test_render_latex_report_mentions_rdma_boundary(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        content = render_latex_report(
            summary_rows=[],
            ps_rows=[],
            gap_rows=[],
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        )

        self.assertIn("RDMA", content)
        self.assertIn("PyTorch/model", content)
        self.assertIn("单机多卡", content)
        self.assertIn("论文实验章节对齐", content)
        self.assertIn("\\documentclass", content)
        self.assertIn("\\begin{document}", content)
        self.assertIn("\\end{document}", content)
        self.assertIn("实验元数据", content)
        self.assertIn("Median M keys/s", render_latex_report(
            summary_rows=[],
            ps_rows=[
                {
                    "transport": "RDMA",
                    "source_profile": "rdma_pet",
                    "phase": "run",
                    "key_ops_per_sec": 7000000,
                    "status": "ok",
                }
            ],
            gap_rows=[],
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        ))

    def test_render_latex_report_includes_ps_failure_rows(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        content = render_latex_report(
            summary_rows=[],
            ps_rows=[
                {
                    "transport": "RDMA",
                    "status": "client_failure",
                    "source_profile": "rdma_pet_matrix_0601_v1024_b1024",
                    "value_size": "1024",
                    "batch_keys": "1024",
                    "message": "client exited with code -6",
                }
            ],
            gap_rows=[],
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        )

        self.assertIn("失败或容量限制如下", content)
        self.assertIn("batch\\_keys=1024: client exited with code -6", content)

    def test_render_latex_report_includes_rdma_client_process_scaling(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        content = render_latex_report(
            summary_rows=[],
            ps_rows=[
                {
                    "transport": "RDMA",
                    "status": "success",
                    "source_profile": "rdma_pet_cp1",
                    "phase": "run",
                    "value_size": "512",
                    "batch_keys": "500",
                    "client_processes": "1",
                    "repeat_index": "0",
                    "key_ops_per_sec": "22000000",
                },
                {
                    "transport": "RDMA",
                    "status": "success",
                    "source_profile": "rdma_pet_cp8",
                    "phase": "run",
                    "value_size": "512",
                    "batch_keys": "500",
                    "client_processes": "8",
                    "repeat_index": "0",
                    "client_index": "0",
                    "key_ops_per_sec": "5600000",
                },
                {
                    "transport": "RDMA",
                    "status": "success",
                    "source_profile": "rdma_pet_cp8",
                    "phase": "run",
                    "value_size": "512",
                    "batch_keys": "500",
                    "client_processes": "8",
                    "repeat_index": "0",
                    "client_index": "1",
                    "key_ops_per_sec": "5400000",
                },
            ],
            gap_rows=[],
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        )

        self.assertIn("RDMA client process 扩展性", content)
        self.assertIn("Client procs", content)
        self.assertIn("Median per-client M keys/s", content)
        self.assertIn("Median total M keys/s", content)
        self.assertIn("1 & 512 & 500 & 22.00 & 22.00", content)
        self.assertIn("8 & 512 & 500 & 5.50 & 11.00", content)

    def test_render_latex_report_does_not_truncate_gap_rows(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        gap_rows = [
            {
                "rows": i + 1,
                "batch_size": 512,
                "num_embeddings": 50000,
                "embedding_dim": 64,
                "recstore_vs_hbm_speedup": 1.0,
                "recstore_vs_uvm_speedup": 1.0,
                "best_recstore_label": f"RecStore-{i + 1}",
            }
            for i in range(35)
        ]

        content = render_latex_report(
            summary_rows=[],
            ps_rows=[],
            gap_rows=gap_rows,
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        )

        self.assertIn("共 35 个", content)
        self.assertIn("RecStore-35", content)

    def test_render_latex_report_includes_gap_group_summary(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        content = render_latex_report(
            summary_rows=[],
            ps_rows=[],
            gap_rows=[
                {
                    "rows": "524288",
                    "batch_size": "1024",
                    "num_embeddings": "800000",
                    "embedding_dim": "128",
                    "recstore_vs_hbm_speedup": "2.0",
                    "recstore_vs_uvm_speedup": "1.5",
                },
                {
                    "rows": "524288",
                    "batch_size": "4096",
                    "num_embeddings": "4000000",
                    "embedding_dim": "128",
                    "recstore_vs_hbm_speedup": "1.0",
                    "recstore_vs_uvm_speedup": "0.5",
                },
            ],
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        )

        self.assertIn("RecStore/TorchRec 分组几何均值", content)
        self.assertIn("batch<=1024", content)
        self.assertIn("batch>=4096", content)
        self.assertIn("emb>=4M", content)
        self.assertIn("Geo RecStore/UVM", content)

    def test_render_latex_report_includes_artifact_source_table(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        content = render_latex_report(
            summary_rows=[
                {
                    "status": "ok",
                    "source_profile": "source_a",
                    "source_root": "/tmp/source_a",
                }
            ],
            ps_rows=[
                {
                    "status": "success",
                    "source_profile": "rdma_source",
                    "source_root": "/tmp/rdma_source",
                }
            ],
            gap_rows=[],
            metadata={
                "gpu_count": 1,
                "rdma_available": True,
                "profile": "smoke",
                "output_root": "/tmp/full_report",
            },
        )

        self.assertIn("Artifact 与 source 清单", content)
        self.assertIn("\\begin{longtable}", content)
        self.assertIn("summary\\_e2e.csv", content)
        self.assertIn("bench\\_e2e\\_report.tex", content)
        self.assertIn("/tmp/source\\_a", content)
        self.assertIn("/tmp/rdma\\_source", content)

    def test_render_latex_report_includes_environment_table(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        content = render_latex_report(
            summary_rows=[],
            ps_rows=[],
            gap_rows=[],
            metadata={
                "gpu_count": 1,
                "rdma_available": True,
                "profile": "smoke",
                "git_commit": "abcdef",
                "nvidia_smi_gpu": "NVIDIA L40S, 46068 MiB, 550.54.15",
                "torch_version": "2.7.0",
                "torch_cuda": "12.4",
            },
        )

        self.assertIn("硬件与软件环境", content)
        self.assertIn("git\\_commit", content)
        self.assertIn("NVIDIA L40S", content)
        self.assertIn("torch\\_version", content)

    def test_render_latex_report_includes_executive_summary(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        content = render_latex_report(
            summary_rows=[],
            ps_rows=[],
            gap_rows=[
                {
                    "rows": "524288",
                    "batch_size": "1024",
                    "num_embeddings": "800000",
                    "embedding_dim": "128",
                    "recstore_vs_hbm_speedup": "2.0",
                    "recstore_vs_uvm_speedup": "1.5",
                },
                {
                    "rows": "524288",
                    "batch_size": "4096",
                    "num_embeddings": "4000000",
                    "embedding_dim": "128",
                    "recstore_vs_hbm_speedup": "1.0",
                    "recstore_vs_uvm_speedup": "0.5",
                },
            ],
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        )

        self.assertIn("结论摘要", content)
        self.assertIn("可配对配置", content)
        self.assertIn("RecStore/HBM", content)
        self.assertIn("RecStore/UVM", content)
        self.assertIn("单机多卡", content)

    def test_render_latex_report_includes_figures(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        rows = [
            {
                "status": "ok",
                "label": "TorchRec-HBM-1proc",
                "backend": "torchrec",
                "torchrec_memory_mode": "hbm",
                "rows": 4096,
                "batch_size": 512,
                "num_embeddings": 50000,
                "embedding_dim": 64,
                "samples_per_sec": 1000.0,
                "mean_step_total_ms": 10.0,
                "nproc_per_node": 1,
            },
            {
                "status": "ok",
                "label": "RecStore-RDMA-PET-1proc",
                "backend": "recstore",
                "ps_type": "RDMA",
                "rows": 4096,
                "batch_size": 512,
                "num_embeddings": 50000,
                "embedding_dim": 64,
                "samples_per_sec": 1500.0,
                "mean_step_total_ms": 7.0,
                "nproc_per_node": 1,
            },
            {
                "status": "ok",
                "label": "RecStore-RDMA-PET-1proc",
                "backend": "recstore",
                "ps_type": "RDMA",
                "rows": 4096,
                "batch_size": 1024,
                "num_embeddings": 50000,
                "embedding_dim": 64,
                "samples_per_sec": 2500.0,
                "mean_step_total_ms": 8.0,
                "nproc_per_node": 1,
            },
        ]

        content = render_latex_report(
            summary_rows=rows,
            ps_rows=[],
            gap_rows=[],
            metadata={"created_at": "2026-06-02", "gpu_count": 1, "rdma_available": True},
        )

        self.assertIn("\\usepackage{graphicx}", content)
        self.assertIn("分场景图形对比", content)
        self.assertIn("图形阅读结论", content)
        self.assertIn("RecStore-RDMA 创新点", content)
        self.assertIn("\\includegraphics", content)
        self.assertIn("RecStore-RDMA-PET-1proc", content)
        self.assertIn("figures/e2e\\_rdma\\_batch.svg", content)

    def test_write_svg_figures_creates_scenario_plots(self) -> None:
        from tools.benchmarks.run_bench_e2e import write_svg_figures

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_svg_figures(
                Path(tmpdir),
                summary_rows=[
                    {
                        "status": "ok",
                        "label": "TorchRec-HBM-1proc",
                        "backend": "torchrec",
                        "rows": "32768",
                        "batch_size": "256",
                        "num_embeddings": "50000",
                        "embedding_dim": "64",
                        "nproc_per_node": "1",
                        "samples_per_sec": "8000",
                    },
                    {
                        "status": "ok",
                        "label": "RecStore-RDMA-PET-1proc",
                        "backend": "recstore",
                        "ps_type": "RDMA",
                        "rows": "32768",
                        "batch_size": "256",
                        "num_embeddings": "50000",
                        "embedding_dim": "64",
                        "nproc_per_node": "1",
                        "samples_per_sec": "20000",
                    },
                ],
                gap_rows=[
                    {
                        "batch_size": "256",
                        "recstore_vs_hbm_speedup": "2.5",
                        "recstore_vs_uvm_speedup": "0",
                    }
                ],
                ps_rows=[],
            )

            names = {path.name for path in paths}
            self.assertIn("speedup_batch.svg", names)
            self.assertNotIn("e2e_batch.svg", names)
            content = (Path(tmpdir) / "figures" / "speedup_batch.svg").read_text(encoding="utf-8")
            self.assertIn("<svg", content)
            self.assertIn("RecStore/HBM", content)

    def test_write_svg_figures_creates_rdma_failure_plot(self) -> None:
        from tools.benchmarks.run_bench_e2e import write_svg_figures

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_svg_figures(
                Path(tmpdir),
                summary_rows=[
                    {
                        "status": "failed",
                        "label": "RecStore-RDMA-PET-1proc",
                        "backend": "recstore",
                        "ps_type": "RDMA",
                        "num_embeddings": "4000000",
                    }
                ],
                gap_rows=[],
                ps_rows=[],
            )

            names = {path.name for path in paths}
            self.assertIn("rdma_failure_capacity.svg", names)

    def test_build_figure_specs_skips_single_point_dimension_curve(self) -> None:
        from tools.benchmarks.run_bench_e2e import build_figure_specs

        rows = [
            {
                "status": "ok",
                "label": "TorchRec-HBM-1proc",
                "backend": "torchrec",
                "rows": "32768",
                "batch_size": "256",
                "num_embeddings": "50000",
                "embedding_dim": "64",
                "nproc_per_node": "1",
                "samples_per_sec": "8000",
            },
            {
                "status": "ok",
                "label": "RecStore-RDMA-PET-1proc",
                "backend": "recstore",
                "ps_type": "RDMA",
                "rows": "32768",
                "batch_size": "256",
                "num_embeddings": "50000",
                "embedding_dim": "64",
                "nproc_per_node": "1",
                "samples_per_sec": "20000",
            },
        ]

        names = {spec.filename for spec in build_figure_specs(rows, [], [])}

        self.assertNotIn("e2e_dim.svg", names)

    def test_render_latex_report_includes_repeat_stability_table(self) -> None:
        from tools.benchmarks.run_bench_e2e import render_latex_report

        summary_rows = [
            {
                "status": "ok",
                "rows": 32768,
                "batch_size": 512,
                "num_embeddings": 200000,
                "embedding_dim": 64,
                "label": "TorchRec-HBM-1proc",
                "samples_per_sec": value,
            }
            for value in (1000.0, 1100.0, 900.0)
        ]

        content = render_latex_report(
            summary_rows=summary_rows,
            ps_rows=[],
            gap_rows=[],
            metadata={"gpu_count": 1, "rdma_available": True, "profile": "smoke"},
        )

        self.assertIn("重复实验稳定性", content)
        self.assertIn("Mean samples/s", content)
        self.assertIn("TorchRec-HBM-1proc", content)

    def test_build_result_insights_summarizes_batch_and_rdma_boundaries(self) -> None:
        from tools.benchmarks.run_bench_e2e import build_result_insights

        insights = build_result_insights(
            summary_rows=[
                {
                    "status": "ok",
                    "backend": "recstore",
                    "prefetch_depth": 0,
                    "rows": 4096,
                    "batch_size": 512,
                    "num_embeddings": 50000,
                    "embedding_dim": 64,
                    "samples_per_sec": 2000.0,
                },
                {
                    "status": "ok",
                    "backend": "recstore",
                    "prefetch_depth": 4,
                    "rows": 4096,
                    "batch_size": 512,
                    "num_embeddings": 50000,
                    "embedding_dim": 64,
                    "samples_per_sec": 500.0,
                },
            ],
            gap_rows=[
                {
                    "rows": 4096,
                    "batch_size": 512,
                    "num_embeddings": 50000,
                    "embedding_dim": 64,
                    "recstore_vs_hbm_speedup": 2.0,
                    "recstore_vs_uvm_speedup": 1.5,
                },
                {
                    "rows": 131072,
                    "batch_size": 4096,
                    "num_embeddings": 800000,
                    "embedding_dim": 128,
                    "recstore_vs_hbm_speedup": 0.9,
                    "recstore_vs_uvm_speedup": 0.7,
                },
            ],
            ps_rows=[
                {
                    "status": "ok",
                    "phase": "run",
                    "transport": "RDMA",
                    "key_ops_per_sec": 4800000,
                }
            ],
            metadata={"gpu_count": 1},
        )

        joined = "\n".join(insights)
        self.assertIn("batch size", joined)
        self.assertIn("prefetch_depth", joined)
        self.assertIn("PS/network", joined)
        self.assertIn("GPU 数不足", joined)

    def test_combine_existing_roots_merges_manifest_and_ps_rows(self) -> None:
        from tools.benchmarks.run_bench_e2e import combine_existing_roots

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root_a = tmp / "bench_e2e_a"
            root_b = tmp / "bench_e2e_b"
            out = tmp / "combined"
            root_a.mkdir()
            root_b.mkdir()
            for root, run_id in ((root_a, "a1"), (root_b, "b1")):
                with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["run_id", "status"])
                    writer.writeheader()
                    writer.writerow({"run_id": run_id, "status": "ok"})
            with (root_a / "summary_ps_network.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["transport", "status"])
                writer.writeheader()
                writer.writerow({"transport": "RDMA", "status": "ok"})

            manifest, ps_rows = combine_existing_roots([root_a, root_b], out)

            self.assertEqual([row["run_id"] for row in manifest], ["a1", "b1"])
            self.assertEqual(len(ps_rows), 1)
            self.assertTrue((out / "manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
