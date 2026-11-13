from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.config import parse_config
from model_zoo.rs_demo.config import populate_default_paths
from model_zoo.rs_demo.config import dump_run_config, validate_dense_cuda_graph, validate_recstore_config


class TestHybridConfig(unittest.TestCase):
    def test_parse_config_echoes_flags_to_attributes(self) -> None:
        cases = [
            (
                [
                    "--embedding-dim",
                    "128",
                    "--dense-arch-layer-sizes",
                    "512,256,128",
                    "--over-arch-layer-sizes",
                    "1024,1024,512,256,1",
                ],
                {
                    "embedding_dim": 128,
                    "dense_arch_layer_sizes": "512,256,128",
                    "over_arch_layer_sizes": "1024,1024,512,256,1",
                },
            ),
            (
                ["--backend", "recstore", "--ps-kv-backend", "hps_rocksdb"],
                {"ps_kv_backend": "hps_rocksdb"},
            ),
            (
                [
                    "--backend",
                    "recstore",
                    "--ps-kv-backend",
                    "recstore_tiered",
                    "--tiered-dram-capacity-multiplier",
                    "0.02",
                ],
                {"tiered_dram_capacity_multiplier": 0.02},
            ),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                cfg = parse_config(args)
                for attr, value in expected.items():
                    self.assertEqual(getattr(cfg, attr), value)

    def test_tiered_cache_policy_accepts_clock(self) -> None:
        args = ["--backend", "recstore", "--ps-kv-backend", "recstore_tiered"]
        cfg = parse_config(args)
        validate_recstore_config(cfg)
        self.assertEqual(cfg.tiered_cache_policy, "clock")
        for raw, expected in (("clock", "clock"), ("none", "none"), ("LRU", "LRU")):
            with self.subTest(policy=raw):
                cfg = parse_config([*args, "--tiered-cache-policy", raw])
                validate_recstore_config(cfg)
                self.assertEqual(cfg.tiered_cache_policy, expected)
        with self.assertRaisesRegex(RuntimeError, "tiered-cache-policy"):
            validate_recstore_config(
                parse_config([*args, "--tiered-cache-policy", "bogus"])
            )

    def test_dense_graph_config_roundtrip_and_supported_paths(self) -> None:
        self.assertFalse(parse_config([]).dense_cuda_graph)
        for args in (
            ["--read-mode", "ps_pipeline"],
            ["--backend", "torchrec"],
        ):
            with self.subTest(args=args):
                cfg = parse_config([*args, "--dense-cuda-graph"])
                validate_dense_cuda_graph(cfg)
                with tempfile.TemporaryDirectory() as directory:
                    path = dump_run_config(cfg, Path(directory) / "worker.json")
                    restored = parse_config(["--run-config-json", str(path)])
                    self.assertTrue(restored.dense_cuda_graph)
                    validate_dense_cuda_graph(restored)
        for args in (
            [],
            ["--read-mode", "bagpipe"],
            ["--read-mode", "ps_pipeline", "--ps-pipeline-early-submit"],
            ["--backend", "hps_torch"],
            ["--backend", "torchrec", "--torchrec-dist-mode", "fair_remote"],
        ):
            with self.subTest(rejected=args):
                with self.assertRaises(RuntimeError):
                    validate_dense_cuda_graph(parse_config([*args, "--dense-cuda-graph"]))

    def test_dependency_split_defaults_roundtrip_and_scope(self) -> None:
        self.assertFalse(parse_config([]).ps_pipeline_dependency_split)
        args = ["--read-mode", "ps_pipeline", "--nnodes", "2", "--steps", "3",
                "--ps-pipeline-dependency-split", "--recstore-runtime-dir", "/tmp/split-config-test"]
        cfg = parse_config(args)
        validate_recstore_config(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = dump_run_config(cfg, Path(directory) / "worker.json")
            restored = parse_config(["--run-config-json", str(path)])
            self.assertTrue(restored.ps_pipeline_dependency_split)
            validate_recstore_config(restored)
        for extra in (["--backend", "torchrec"], ["--read-mode", "bagpipe"],
                      ["--nnodes", "1"], ["--nnodes", "3"], ["--steps", "0"]):
            with self.subTest(extra=extra), self.assertRaisesRegex(RuntimeError, "dependency-split"):
                validate_recstore_config(parse_config(args + extra))

    def test_populate_default_paths_makes_relative_output_root_absolute(self) -> None:
        cfg = parse_config(
            [
                "--output-root",
                "relative-output",
                "--run-id",
                "case-relative-output",
            ]
        )

        populate_default_paths(cfg)

        self.assertTrue(Path(cfg.output_root).is_absolute())
        self.assertTrue(Path(cfg.recstore_main_csv).is_absolute())
        self.assertIn("relative-output", cfg.recstore_main_csv)


if __name__ == "__main__":
    unittest.main()
