import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps_test_config import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    resolve_rdma_integration_config,
)
from run_petps_integration import normalize_timeout
from run_petps_server import infer_server_count


class TestRunPetPSIntegration(unittest.TestCase):
    def test_timeout_keeps_user_value(self):
        self.assertEqual(normalize_timeout(99, "client-timeout"), 99)

    def test_timeout_keeps_value_within_limit(self):
        self.assertEqual(normalize_timeout(10, "cluster-timeout"), 10)

    def test_timeout_must_be_positive(self):
        with self.assertRaises(ValueError):
            normalize_timeout(0, "client-timeout")

    def test_uses_single_shard_rdma_config_by_default(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=1, config_path=None),
            DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        )

    def test_uses_multi_shard_rdma_config_for_multi_server_runs(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path=None),
            DEFAULT_RDMA_MULTI_SHARD_CONFIG,
        )

    def test_explicit_config_path_wins(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path="./custom.json"),
            "./custom.json",
        )

    def test_petps_server_count_requires_explicit_rdma_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                '{"distributed_client": {"num_shards": 2}}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "rdma_deployment"):
                infer_server_count(config_path)

    def test_petps_server_count_uses_explicit_num_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                '{"distributed_client": {"num_shards": 2}, '
                '"rdma_deployment": {"deployment_id": "test", "epoch": 1, "protocol_version": 1, "num_clients": 1, '
                '"nodes": [{"node_id": 0, "role": "server"}]}}',
                encoding="utf-8",
            )
            self.assertEqual(infer_server_count(config_path), 2)


if __name__ == "__main__":
    unittest.main()
