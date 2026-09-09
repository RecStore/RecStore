#!/usr/bin/env python3

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.recstore_config_path import resolve_recstore_config_path

DEFAULT_GRPC_MAIN_CONFIG = "./recstore_config.json"
DEFAULT_BRPC_BENCHMARK_CONFIG = "./src/test/configs/recstore_config.brpc.json"
DEFAULT_RDMA_SINGLE_SHARD_CONFIG = "./src/test/configs/recstore_config.rdma_test.json"
DEFAULT_RDMA_MULTI_SHARD_CONFIG = "./src/test/configs/recstore_config.rdma_multishard_test.json"


def resolve_repo_path(config_path):
    resolved = Path(config_path)
    if str(config_path) == DEFAULT_GRPC_MAIN_CONFIG:
        return resolve_recstore_config_path()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def resolve_rdma_integration_config(server_count, config_path):
    if config_path:
        return config_path
    if server_count > 1:
        return DEFAULT_RDMA_MULTI_SHARD_CONFIG
    return DEFAULT_RDMA_SINGLE_SHARD_CONFIG


def materialize_rdma_config(config_path, server_count, client_count, runtime_dir=None):
    """Write a runtime RDMA config whose deployment matches the launched mesh."""
    if server_count <= 0 or client_count <= 0:
        raise ValueError("server_count and client_count must be positive")
    with resolve_repo_path(config_path).open() as fh:
        config = json.load(fh)
    distributed = config.get("distributed_client")
    deployment = config.get("rdma_deployment")
    if not isinstance(distributed, dict) or not isinstance(deployment, dict):
        raise ValueError("RDMA config requires distributed_client and rdma_deployment")
    if distributed.get("num_shards") != server_count:
        raise ValueError("distributed_client.num_shards does not match server_count")
    nodes = deployment.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("rdma_deployment.nodes must be an array")
    server_nodes = [node for node in nodes if node.get("role") == "server"]
    client_nodes = [node for node in nodes if node.get("role") == "client"]
    if len(server_nodes) != server_count:
        raise ValueError("rdma_deployment server node count does not match server_count")
    if [node.get("node_id") for node in server_nodes] != list(range(server_count)):
        raise ValueError("rdma_deployment server node ids must be dense")
    if not client_nodes:
        raise ValueError("rdma_deployment requires a client fabric node")
    deployment["num_clients"] = client_count
    deployment["nodes"] = server_nodes + [
        {**client_nodes[0], "node_id": server_count + client_index, "role": "client"}
        for client_index in range(client_count)
    ]
    if runtime_dir is None:
        runtime_dir = tempfile.mkdtemp(prefix="recstore_rdma_config_")
    runtime_path = Path(runtime_dir) / "recstore_config.rdma_runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return str(runtime_path)


def load_client_endpoint(config_path):
    with resolve_repo_path(config_path).open() as fh:
        config = json.load(fh)
    client = config["client"]
    return client["host"], client["port"]
