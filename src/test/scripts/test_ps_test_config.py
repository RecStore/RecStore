import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools.config.recstore_config_path as recstore_config_path
from ps_test_config import (
    DEFAULT_BRPC_BENCHMARK_CONFIG,
    DEFAULT_GRPC_MAIN_CONFIG,
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    load_client_endpoint,
    materialize_rdma_config,
    resolve_rdma_integration_config,
    resolve_repo_path,
)


class TestPSTestConfig(unittest.TestCase):
    def test_rdma_config_roles_are_explicit(self):
        self.assertEqual(
            DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
            "./src/test/configs/recstore_config.rdma_test.json",
        )
        self.assertEqual(
            DEFAULT_RDMA_MULTI_SHARD_CONFIG,
            "./src/test/configs/recstore_config.rdma_multishard_test.json",
        )

    def test_default_non_rdma_config_roles_are_explicit(self):
        self.assertEqual(DEFAULT_GRPC_MAIN_CONFIG, "./recstore_config.json")
        self.assertEqual(
            DEFAULT_BRPC_BENCHMARK_CONFIG,
            "./src/test/configs/recstore_config.brpc.json",
        )

    def test_default_grpc_config_resolves_from_search_path_before_default(self):
        with tempfile.TemporaryDirectory() as default_tmp, tempfile.TemporaryDirectory() as cwd_tmp:
            default_config = Path(default_tmp) / "recstore_config.json"
            search_config = Path(cwd_tmp) / "repo" / "recstore_config.json"
            nested = search_config.parent / "a" / "b"
            nested.mkdir(parents=True)
            default_config.write_text("{}", encoding="utf-8")
            search_config.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                recstore_config_path,
                "DEFAULT_RECSTORE_CONFIG_PATH",
                default_config,
            ), mock.patch("pathlib.Path.cwd", return_value=nested):
                self.assertEqual(
                    resolve_repo_path(DEFAULT_GRPC_MAIN_CONFIG),
                    search_config.resolve(),
                )

    def test_resolve_rdma_integration_config_prefers_explicit_path(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path="./custom.json"),
            "./custom.json",
        )

    def test_resolve_rdma_integration_config_uses_single_shard_default(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=1, config_path=None),
            DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        )

    def test_resolve_rdma_integration_config_uses_multi_shard_default(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path=None),
            DEFAULT_RDMA_MULTI_SHARD_CONFIG,
        )

    def test_default_rdma_configs_have_explicit_deployment_nodes(self):
        for config_path, expected_shards in (
            (DEFAULT_RDMA_SINGLE_SHARD_CONFIG, 1),
            (DEFAULT_RDMA_MULTI_SHARD_CONFIG, 2),
        ):
            config = json.loads(resolve_repo_path(config_path).read_text())
            deployment = config["rdma_deployment"]
            self.assertTrue(deployment["deployment_id"])
            self.assertGreater(deployment["epoch"], 0)
            self.assertEqual(deployment["protocol_version"], 1)
            self.assertEqual(config["distributed_client"]["num_shards"], expected_shards)
            self.assertEqual(deployment["num_clients"], 1)
            nodes = deployment["nodes"]
            self.assertEqual(len(nodes), expected_shards + deployment["num_clients"])
            self.assertEqual(
                {node["node_id"] for node in nodes},
                set(range(len(nodes))),
            )
            for node in nodes:
                self.assertIn(node["role"], {"server", "client"})
                self.assertTrue(node["device"])
                self.assertGreater(node["port"], 0)
                self.assertGreaterEqual(node["gid_index"], 0)
                self.assertIn(node["mode"], {"ib", "rocev1", "rocev2"})

    def test_load_client_endpoint_for_default_grpc_config(self):
        host, port = load_client_endpoint(DEFAULT_GRPC_MAIN_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 15000)

    def test_materialize_rdma_config_expands_client_nodes(self):
        with tempfile.TemporaryDirectory() as runtime_dir:
            path = materialize_rdma_config(
                DEFAULT_RDMA_SINGLE_SHARD_CONFIG, 1, 3, runtime_dir
            )
            config = json.loads(Path(path).read_text())
        deployment = config["rdma_deployment"]
        self.assertEqual(deployment["num_clients"], 3)
        self.assertEqual(
            [(node["node_id"], node["role"]) for node in deployment["nodes"]],
            [(0, "server"), (1, "client"), (2, "client"), (3, "client")],
        )

    def test_materialize_rdma_config_rejects_non_dense_servers(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as runtime_dir:
            source = Path(source_dir) / "config.json"
            config = json.loads(Path(resolve_repo_path(DEFAULT_RDMA_SINGLE_SHARD_CONFIG)).read_text())
            config["rdma_deployment"]["nodes"][0]["node_id"] = 2
            source.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "dense"):
                materialize_rdma_config(str(source), 1, 1, runtime_dir)

    def test_load_client_endpoint_for_brpc_benchmark_config(self):
        host, port = load_client_endpoint(DEFAULT_BRPC_BENCHMARK_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 25000)


if __name__ == "__main__":
    unittest.main()
