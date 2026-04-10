from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.runtime.server import (
    choose_available_ports,
    make_runtime_dir,
    resolve_kv_data_path,
)


class TestChooseAvailablePorts(unittest.TestCase):
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
        base_cfg = {"cache_ps": {}, "distributed_client": {"servers": []}}
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir, runtime_cfg_path = make_runtime_dir(
                base_cfg=base_cfg,
                host="127.0.0.1",
                port0=15123,
                port1=15124,
                allocator="PersistLoopShmMalloc",
                output_root=tmpdir,
                run_id="case-a",
                ps_type="BRPC",
            )
            self.assertTrue(str(runtime_dir).startswith(f"{tmpdir}/runtime/case-a"))
            self.assertEqual(runtime_cfg_path, runtime_dir / "recstore_config.json")
            self.assertTrue(runtime_cfg_path.exists())
            runtime_cfg = runtime_cfg_path.read_text(encoding="utf-8")
            self.assertIn(str(Path(tmpdir) / "runtime" / "case-a"), runtime_cfg)

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
            runtime_dir, runtime_cfg_path = make_runtime_dir(
                base_cfg=base_cfg,
                host="127.0.0.1",
                port0=15123,
                port1=15124,
                allocator="PersistLoopShmMalloc",
                output_root=tmpdir,
                run_id="case-cap",
                ps_type="BRPC",
                kv_capacity=520_000,
            )
            self.assertTrue(str(runtime_dir).startswith(f"{tmpdir}/runtime/case-cap"))
            runtime_cfg = runtime_cfg_path.read_text(encoding="utf-8")
            self.assertIn('"capacity": 520000', runtime_cfg)

    def test_make_runtime_dir_keeps_shared_base_kv_prefix_for_sharded_server(self) -> None:
        base_cfg = {"cache_ps": {}, "distributed_client": {"servers": []}}
        with tempfile.TemporaryDirectory() as tmpdir:
            _runtime_dir, runtime_cfg_path = make_runtime_dir(
                base_cfg=base_cfg,
                host="127.0.0.1",
                port0=15123,
                port1=15124,
                allocator="PersistLoopShmMalloc",
                output_root=tmpdir,
                run_id="case-shards",
                ps_type="BRPC",
            )
            runtime_cfg = runtime_cfg_path.read_text(encoding="utf-8")
            self.assertIn('"path"', runtime_cfg)
            self.assertIn(str(Path(tmpdir) / "runtime" / "case-shards"), runtime_cfg)

    def test_r2shmmalloc_kv_path_falls_back_to_local_tmp(self) -> None:
        path = resolve_kv_data_path(
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="case-r2",
            path_suffix="abc123",
            allocator="R2ShmMalloc",
        )
        self.assertEqual(path, "/tmp/rs_demo_kv/case-r2/kv_abc123")


if __name__ == "__main__":
    unittest.main()
