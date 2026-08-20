from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from python.pytorch.recstore.benchmark.server import (
    choose_available_ports,
    make_runtime_dir,
    resolve_kv_data_path,
    wait_server_ready,
)


class TestChooseAvailablePorts(unittest.TestCase):
    def _make_runtime_dir(
        self,
        output_root,
        *,
        base_cfg=None,
        run_id="case",
        ps_type="BRPC",
        **kwargs,
    ):
        if base_cfg is None:
            base_cfg = {"cache_ps": {}, "distributed_client": {"servers": []}}
        return make_runtime_dir(
            base_cfg=base_cfg,
            host="127.0.0.1",
            port0=15123,
            port1=15124,
            allocator="PersistLoopShmMalloc",
            output_root=output_root,
            run_id=run_id,
            ps_type=ps_type,
            **kwargs,
        )

    def test_return_preferred_when_free(self) -> None:
        with socket.socket() as s0, socket.socket() as s1:
            s0.bind(("127.0.0.1", 0))
            s1.bind(("127.0.0.1", 0))
            p0 = s0.getsockname()[1]
            p1 = s1.getsockname()[1]

        got0, got1 = choose_available_ports("127.0.0.1", p0, p1)
        self.assertEqual((got0, got1), (p0, p1))

    def test_fallback_when_preferred_busy(self) -> None:
        with socket.socket() as s0, socket.socket() as s1:
            s0.bind(("127.0.0.1", 0))
            p0 = s0.getsockname()[1]
            s1.bind(("127.0.0.1", p0 + 1))
            p1 = s1.getsockname()[1]

            got0, got1 = choose_available_ports("127.0.0.1", p0, p1)
            self.assertNotEqual((got0, got1), (p0, p1))
            self.assertNotEqual(got0, got1)

    def test_make_runtime_dir_uses_output_root_and_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir, runtime_cfg_path = self._make_runtime_dir(tmpdir, run_id="case-a")
            self.assertTrue(str(runtime_dir).startswith(f"{tmpdir}/runtime/case-a"))
            self.assertEqual(runtime_cfg_path, runtime_dir / "recstore_config.json")
            self.assertTrue(runtime_cfg_path.exists())
            runtime_cfg = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))
            self.assertTrue(
                runtime_cfg["cache_ps"]["base_kv_config"]["value"]["path"].startswith(
                    "/dev/shm/rs_demo_kv/case-a/"
                )
            )

    def test_make_runtime_dir_returns_absolute_paths_for_relative_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                    "relative-output", run_id="case-relative"
                )
            finally:
                os.chdir(old_cwd)

            self.assertTrue(runtime_dir.is_absolute())
            self.assertTrue(runtime_cfg_path.is_absolute())
            self.assertTrue(runtime_cfg_path.exists())

    def test_make_runtime_dir_overrides_kv_capacity_when_requested(self) -> None:
        base_cfg = {
            "cache_ps": {
                "base_kv_config": {
                    "capacity": 8_000_000,
                }
            },
            "distributed_client": {"servers": []},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                tmpdir, base_cfg=base_cfg, run_id="case-cap", kv_capacity=520_000
            )
            self.assertTrue(str(runtime_dir).startswith(f"{tmpdir}/runtime/case-cap"))
            runtime_cfg = runtime_cfg_path.read_text(encoding="utf-8")
            self.assertIn('"capacity": 520000', runtime_cfg)

    def test_make_runtime_dir_writes_local_shm_runtime_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                tmpdir,
                run_id="case-local-shm",
                ps_type="LOCAL_SHM",
                value_size_bytes=256,
            )
            runtime_cfg = runtime_cfg_path.read_text(encoding="utf-8")
            self.assertIn('"ps_type": "LOCAL_SHM"', runtime_cfg)
            self.assertIn('"local_shm"', runtime_cfg)
            self.assertIn('"region_name"', runtime_cfg)
            self.assertIn('"default_value_size_hint": 256', runtime_cfg)

    def test_make_runtime_dir_writes_tiered_base_kv_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                tmpdir,
                run_id="case-tiered",
                ps_type="GRPC",
                kv_capacity=20_000,
                value_size_bytes=512,
                ps_kv_backend="recstore_tiered",
            )
            runtime_cfg = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))
            base_kv = runtime_cfg["cache_ps"]["base_kv_config"]

            self.assertEqual(base_kv["value"]["type"], "TIERED_VALUE_STORE")
            self.assertNotIn("path", base_kv["value"])
            self.assertTrue(
                base_kv["value"]["dram_allocator"]["path"].startswith(
                    "/dev/shm/rs_demo_kv/case-tiered/"
                )
            )
            self.assertIn("/tmp/rs_demo_kv/case-tiered/", base_kv["value"]["ssd_allocator"]["path"])
            self.assertGreaterEqual(
                base_kv["value"]["ssd_allocator"]["capacity_bytes"],
                256 * 1024 * 1024,
            )
            self.assertEqual(base_kv["value"]["tiering"], {"cache_policy": "LRU"})

    def test_make_runtime_dir_applies_tiered_dram_capacity_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                tmpdir,
                run_id="case-tiered-small-dram",
                ps_type="GRPC",
                kv_capacity=20_000,
                value_size_bytes=512,
                ps_kv_backend="recstore_tiered",
                tiered_dram_capacity_multiplier=0.02,
            )
            runtime_cfg = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))
            base_kv = runtime_cfg["cache_ps"]["base_kv_config"]

            self.assertEqual(
                base_kv["value"]["dram_allocator"]["capacity_bytes"],
                int(20_000 * 512 * 0.02),
            )

    def test_make_runtime_dir_writes_hps_rocksdb_base_kv_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                tmpdir,
                run_id="case-hps-rocks",
                ps_type="GRPC",
                kv_capacity=20_000,
                value_size_bytes=512,
                ps_kv_backend="hps_rocksdb",
            )
            runtime_cfg = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))
            base_kv = runtime_cfg["cache_ps"]["base_kv_config"]

            self.assertEqual(base_kv["external_engine_type"], "KVEngineHPSRocksDB")
            self.assertEqual(base_kv["capacity"], 20_000)
            self.assertEqual(base_kv["value_size"], 512)
            self.assertEqual(base_kv["rocksdb_path"], base_kv["path"])
            self.assertNotIn("index", base_kv)
            self.assertNotIn("value", base_kv)
            self.assertTrue(Path(base_kv["path"]).exists())

    def test_make_runtime_dir_writes_hps_hash_map_base_kv_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                tmpdir,
                run_id="case-hps-hash",
                ps_type="GRPC",
                kv_capacity=20_000,
                value_size_bytes=512,
                ps_kv_backend="hps_hash_map",
            )
            runtime_cfg = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))
            base_kv = runtime_cfg["cache_ps"]["base_kv_config"]

            self.assertEqual(base_kv["external_engine_type"], "KVEngineHPSHashMap")
            self.assertEqual(base_kv["capacity"], 20_000)
            self.assertEqual(base_kv["value_size"], 512)
            self.assertNotIn("rocksdb_path", base_kv)
            self.assertNotIn("index", base_kv)
            self.assertNotIn("value", base_kv)

    def test_make_runtime_dir_uses_single_shard_for_e2e_bringup(self) -> None:
        for ps_type, run_id in (("LOCAL_SHM", "case-local-shm-single"), ("RDMA", "case-rdma-single")):
            with self.subTest(ps_type=ps_type):
                with tempfile.TemporaryDirectory() as tmpdir:
                    _runtime_dir, runtime_cfg_path = self._make_runtime_dir(
                        tmpdir,
                        run_id=run_id,
                        ps_type=ps_type,
                        value_size_bytes=256,
                    )
                    runtime_cfg = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))

                    self.assertEqual(runtime_cfg["cache_ps"]["num_shards"], 1)
                    self.assertEqual(runtime_cfg["distributed_client"]["num_shards"], 1)
                    self.assertEqual(runtime_cfg["cache_ps"]["servers"], [{"host": "127.0.0.1", "port": 15123, "shard": 0}])
                    self.assertEqual(runtime_cfg["distributed_client"]["servers"], [{"host": "127.0.0.1", "port": 15123, "shard": 0}])

    def test_wait_server_ready_local_shm_only_requires_live_process(self) -> None:
        proc = mock.Mock()
        proc.poll.return_value = None
        self.assertTrue(
            wait_server_ready(
                proc=proc,
                host="127.0.0.1",
                port0=15123,
                port1=15124,
                timeout_s=0.1,
                ps_type="LOCAL_SHM",
            )
        )

    def test_dram_kv_path_uses_dev_shm_for_backend_policy(self) -> None:
        path = resolve_kv_data_path(
            output_root="/tmp/rs_demo",
            run_id="case-r2",
            path_suffix="abc123",
            allocator="R2ShmMalloc",
        )
        self.assertEqual(path, "/dev/shm/rs_demo_kv/case-r2/kv_abc123")


if __name__ == "__main__":
    unittest.main()
