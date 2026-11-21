import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.benchmarks.e2e.custom.config import (  # noqa: E402
    BenchmarkConfig,
    ClientSpec,
    ServerSpec,
)
from tools.benchmarks.e2e.custom import runtime  # noqa: E402

FABRIC_KEYS = (
    "RS_DEMO_RDMA_FABRIC",
    "RS_DEMO_RDMA_FABRIC_SERVER",
    "RS_DEMO_RDMA_FABRIC_CLIENT",
)


class TestRdmaFabricSpec(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in FABRIC_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        self._env.stop()

    def test_shared_key_applies_to_both_roles(self):
        os.environ["RS_DEMO_RDMA_FABRIC"] = "mlx5_0:1:0:ib"
        self.assertEqual(runtime._rdma_fabric_node(0, "server")["device"], "mlx5_0")
        self.assertEqual(runtime._rdma_fabric_node(1, "client")["device"], "mlx5_0")

    def test_role_key_overrides_shared_key(self):
        os.environ["RS_DEMO_RDMA_FABRIC"] = "mlx5_0:1:0:ib"
        os.environ["RS_DEMO_RDMA_FABRIC_SERVER"] = "ibp139s0:1:0:ib"
        self.assertEqual(runtime._rdma_fabric_node(0, "server")["device"], "ibp139s0")
        self.assertEqual(runtime._rdma_fabric_node(1, "client")["device"], "mlx5_0")

    def test_role_keys_work_without_shared_key(self):
        os.environ["RS_DEMO_RDMA_FABRIC_SERVER"] = "ibp139s0:1:0:ib"
        os.environ["RS_DEMO_RDMA_FABRIC_CLIENT"] = "mlx5_0:1:0:ib"
        server = runtime._rdma_fabric_node(0, "server")
        client = runtime._rdma_fabric_node(1, "client")
        self.assertEqual((server["device"], server["mode"]), ("ibp139s0", "ib"))
        self.assertEqual((client["device"], client["mode"]), ("mlx5_0", "ib"))

    def test_malformed_spec_is_rejected(self):
        os.environ["RS_DEMO_RDMA_FABRIC_SERVER"] = "ibp139s0:1"
        with self.assertRaises(ValueError):
            runtime._rdma_fabric_node(0, "server")

    def test_runtime_config_wires_role_devices(self):
        os.environ["RS_DEMO_RDMA_FABRIC_SERVER"] = "ibp139s0:1:0:ib"
        os.environ["RS_DEMO_RDMA_FABRIC_CLIENT"] = "mlx5_0:1:0:ib"
        cfg = BenchmarkConfig(
            clients=(ClientSpec(ip="10.0.2.191"),),
            servers=(ServerSpec(ip="10.0.2.190"),),
        )
        config = runtime.build_runtime_config(
            cfg, transport="RDMA", value_path=Path("/tmp/value")
        )
        devices = {
            node["role"]: node["device"] for node in config["rdma_deployment"]["nodes"]
        }
        self.assertEqual(devices, {"server": "ibp139s0", "client": "mlx5_0"})


if __name__ == "__main__":
    unittest.main()
