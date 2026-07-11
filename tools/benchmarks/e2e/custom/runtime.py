from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ..commands import wrap_remote_command
from ..common import SPARSE_FEATURES_PER_SAMPLE, _dense_arch_for_embedding_dim
from .config import BenchmarkConfig, ClientSpec, ServerSpec


def _server_entries(servers: tuple[ServerSpec, ...]) -> list[dict[str, Any]]:
    return [
        {"host": server.ip, "port": int(server.port), "shard": int(server.shard_id)}
        for server in sorted(servers, key=lambda item: item.shard_id)
    ]


def estimate_runtime_capacity(num_embeddings: int, init_rows: int) -> int:
    # DLRM uses one sparse id per table, so the PS key space spans all tables.
    per_table_rows = max(int(num_embeddings), int(init_rows))
    return max(per_table_rows * SPARSE_FEATURES_PER_SAMPLE * 2, 100_000)


def build_runtime_config(cfg: BenchmarkConfig, *, transport: str, value_path: Path) -> dict[str, Any]:
    transport_upper = transport.upper()
    if transport_upper not in {"BRPC", "GRPC", "RDMA"}:
        raise ValueError(f"unsupported E2E transport: {transport}")
    value_size = int(cfg.embedding_dim) * 4
    servers = _server_entries(cfg.servers)
    capacity = estimate_runtime_capacity(cfg.num_embeddings, cfg.init_rows)
    capacity_bytes = capacity * value_size * 2
    base_kv = {
        "capacity": capacity,
        "index": {"type": cfg.index_type},
        "value": {
            "type": "DRAM_VALUE_STORE",
            "path": str(value_path),
            "default_value_size_hint": value_size,
            "dram_allocator": {
                "type": "PERSIST_LOOP_SLAB",
                "capacity_bytes": capacity_bytes,
            },
        },
    }
    return {
        "cache_ps": {
            "ps_type": transport_upper,
            "max_batch_keys_size": 65536,
            "num_threads": 32,
            "num_shards": len(servers),
            "servers": servers,
            "base_kv_config": base_kv,
        },
        "distributed_client": {
            "num_shards": len(servers),
            "hash_method": "city_hash",
            "max_keys_per_request": 65536,
            "servers": servers,
        },
        "client": servers[0],
    }


def _wrap_remote(cmd: list[str], *, ssh_host: str, ssh_port: int, cwd: Path) -> list[str]:
    if ssh_host in {"", "local", "localhost"}:
        return cmd
    return wrap_remote_command(cmd, ssh_host, cwd=cwd, ssh_port=ssh_port)


def _cuda_visible_devices(client: ClientSpec) -> str:
    start = int(client.gpu_id)
    count = max(int(client.nproc_per_node), 1)
    if count <= 1:
        return str(start)
    return ",".join(str(start + offset) for offset in range(count))


def _client_node_count(clients: tuple[ClientSpec, ...]) -> int:
    return max((client.node_rank for client in clients), default=0) + 1


def _rdma_client_process_count(clients: tuple[ClientSpec, ...]) -> int:
    return sum(max(int(client.nproc_per_node), 1) for client in clients)


def _rdma_client_env(runner: Any) -> dict[str, str]:
    env = {
        "RECSTORE_RDMA_RC_NAMESPACE": str(runner.rdma_namespace),
        "RECSTORE_RDMA_CONTROL_PLANE_HOST": str(runner.rdma_control_plane_host),
        "RECSTORE_RDMA_CONTROL_PLANE_PORT": str(runner.rdma_control_plane_port),
    }
    if runner.rdma_control_plane_timeout_ms is not None:
        env["RECSTORE_RDMA_CONTROL_PLANE_TIMEOUT_MS"] = str(
            runner.rdma_control_plane_timeout_ms
        )
    if runner.rdma_wait_timeout_ms is not None:
        env["RECSTORE_RDMA_WAIT_TIMEOUT_MS"] = str(runner.rdma_wait_timeout_ms)
    if runner.rdma_qps_per_client_per_shard is not None:
        env["RECSTORE_RDMA_RC_QPS_PER_CLIENT_PER_SHARD"] = str(
            runner.rdma_qps_per_client_per_shard
        )
    if runner.rdma_slots_per_qp is not None:
        env["RECSTORE_RDMA_RC_SLOTS_PER_QP"] = str(runner.rdma_slots_per_qp)
    if runner.rdma_server_coroutines_per_thread is not None:
        env["RECSTORE_RDMA_RC_SERVER_COROUTINES_PER_THREAD"] = str(
            runner.rdma_server_coroutines_per_thread
        )
    if runner.rdma_server_get_workers is not None:
        env["RECSTORE_RDMA_RC_SERVER_GET_WORKERS"] = str(runner.rdma_server_get_workers)
    return env


def start_rdma_ps_cluster(
    *,
    cfg: BenchmarkConfig,
    config_path: Path,
    log_path: Path,
    control_plane_host: str,
) -> Any:
    sorted_servers = sorted(cfg.servers, key=lambda item: item.shard_id)
    repo_root = sorted_servers[0].repo_root
    scripts_dir = repo_root / "src/test/scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from petps_cluster_runner import PetPSClusterRunner  # type: ignore

    has_remote_server = any(
        server.ssh_host not in {"", "local", "localhost"} for server in sorted_servers
    )

    def wrap_server_command(global_id: int, cmd: list[str]) -> list[str]:
        server = sorted_servers[global_id]
        return _wrap_remote(
            cmd,
            ssh_host=server.ssh_host,
            ssh_port=server.ssh_port,
            cwd=server.repo_root,
        )

    value_size = int(cfg.embedding_dim) * 4
    max_kv_num_per_request = max(1, int(cfg.batch_size) * SPARSE_FEATURES_PER_SAMPLE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runner = PetPSClusterRunner(
        server_path=str(repo_root / "build/bin/petps_server"),
        config_path=str(config_path),
        num_servers=len(cfg.servers),
        num_clients=_rdma_client_process_count(cfg.clients),
        thread_num=16,
        value_size=value_size,
        max_kv_num_per_request=max_kv_num_per_request,
        timeout=180,
        startup_delay=0.0,
        log_dir=str(log_path.parent),
        verbose=True,
        show_status_logs=True,
        show_control_plane_logs=True,
        rdma_namespace=f"rs-e2e-{cfg.output_dir.name}",
        rdma_control_plane_host=control_plane_host,
        rdma_wait_timeout_ms=120000,
        rdma_control_plane_timeout_ms=300000,
        rdma_qps_per_client_per_shard=32,
        rdma_slots_per_qp=1,
        rdma_server_coroutines_per_thread=1,
        rdma_server_get_workers=0,
        rdma_profile_interval_ms=int(
            os.getenv("RECSTORE_E2E_RDMA_PROFILE_INTERVAL_MS", "1000")
        ),
        server_command_wrapper=wrap_server_command if has_remote_server else None,
    )
    with log_path.open("w", encoding="utf-8") as f:
        f.write(
            "petps_server cluster\n"
            f"config_path={config_path}\n"
            f"namespace={runner.rdma_namespace}\n"
            f"control_plane={runner.rdma_control_plane_host}:{runner.rdma_control_plane_port}\n"
            f"num_clients={runner.num_clients}\n"
        )
    runner._rs_demo_log_path = log_path  # type: ignore[attr-defined]
    runner.start()
    return runner


def stop_rdma_ps_cluster(runner: Any) -> None:
    if runner is not None:
        runner.stop()


def build_server_command(*, server: ServerSpec, runtime_config: Path, transport: str) -> list[str]:
    if transport.upper() == "BRPC":
        cmd = [
            "env",
            *(f"{key}={value}" for key, value in _brpc_rdma_env().items()),
            str(server.repo_root / "build/bin/ps_server"),
            "--config_path",
            str(runtime_config),
            "--brpc_server_port",
            str(server.port),
        ]
    else:
        cmd = [str(server.repo_root / "build/bin/ps_server"), "--config_path", str(runtime_config)]
    return _wrap_remote(cmd, ssh_host=server.ssh_host, ssh_port=server.ssh_port, cwd=server.repo_root)


def _nccl_socket_ifnames() -> str:
    # 192 uses enp3s0f0 for 10.0.2.192; 191 uses eno8303 for 10.0.2.191.
    # NCCL/GLOO match by subnet, so listing both is safe on either host and
    # avoids picking a docker/flannel interface (which crashed earlier runs
    # with "socketFinalizeAccept: wrong type 4 != 3").
    return "enp3s0f0,eno8303"


def _recstore_nccl_env() -> dict[str, str]:
    # RecStore lanes only use NCCL for the small dense all-reduce; the embedding
    # traffic is the PS transport under test (BRPC-RDMA or raw-verbs RDMA), so
    # keep NCCL on TCP sockets over the Ethernet control net.
    ifnames = _nccl_socket_ifnames()
    return {
        "NCCL_SOCKET_IFNAME": ifnames,
        "GLOO_SOCKET_IFNAME": ifnames,
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_IB_DISABLE": "1",
    }


def _brpc_rdma_env() -> dict[str, str]:
    # Patched RecStore brpc client/server read these to enable RDMA over mlx5_0.
    return {
        "RECSTORE_BRPC_USE_RDMA": "1",
        "RECSTORE_BRPC_RDMA_DEVICE": "mlx5_0",
    }


def _torchrec_nccl_env() -> dict[str, str]:
    # TorchRec's embedding all-reduce IS the traffic we want on the IB NIC.
    ifnames = _nccl_socket_ifnames()
    return {
        "NCCL_SOCKET_IFNAME": ifnames,
        "GLOO_SOCKET_IFNAME": ifnames,
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": "mlx5_0",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "NET",
    }


def build_client_command(
    *,
    cfg: BenchmarkConfig,
    transport: str,
    client: ClientSpec,
    run_id: str,
    rdzv_id: str | None = None,
    rdma_runner: Any | None = None,
) -> list[str]:
    first_server = sorted(cfg.servers, key=lambda item: item.shard_id)[0]
    env_prefix = ["env", f"CUDA_VISIBLE_DEVICES={_cuda_visible_devices(client)}"]
    env_prefix.extend(f"{key}={value}" for key, value in _recstore_nccl_env().items())
    if transport.upper() == "BRPC":
        env_prefix.extend(f"{key}={value}" for key, value in _brpc_rdma_env().items())
    if transport.upper() == "RDMA" and rdma_runner is not None:
        env_prefix.extend(
            f"{key}={value}" for key, value in _rdma_client_env(rdma_runner).items()
        )
    cmd = [
        *env_prefix,
        cfg.python_bin,
        str(client.repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        "recstore",
        "--ps-type",
        transport.upper(),
        "--recstore-index-type",
        cfg.index_type,
        "--ps-kv-backend",
        "recstore_dram",
        "--batch-size",
        str(cfg.batch_size),
        "--embedding-dim",
        str(cfg.embedding_dim),
        "--num-embeddings",
        str(cfg.num_embeddings),
        "--init-rows",
        str(cfg.init_rows),
        "--steps",
        str(cfg.steps),
        "--warmup-steps",
        str(cfg.warmup_steps),
        "--read-mode",
        cfg.read_mode,
        "--prefetch-depth",
        str(cfg.prefetch_depth),
        "--dense-arch-layer-sizes",
        _dense_arch_for_embedding_dim(cfg.embedding_dim),
        "--data-dir",
        str(cfg.dataset_path),
        "--output-root",
        str(cfg.output_dir),
        "--run-id",
        run_id,
        "--recstore-runtime-dir",
        str(cfg.resolved_runtime_dir / transport.lower()),
        "--no-start-server",
        "--server-host",
        first_server.ip,
        "--server-port0",
        str(first_server.port),
        "--nnodes",
        str(_client_node_count(cfg.clients)),
        "--node-rank",
        str(client.node_rank),
        "--nproc-per-node",
        str(client.nproc_per_node),
        "--master-addr",
        cfg.clients[0].ip,
        "--master-port",
        str(cfg.master_port),
        "--rdzv-id",
        rdzv_id or run_id,
    ]
    return _wrap_remote(cmd, ssh_host=client.ssh_host, ssh_port=client.ssh_port, cwd=client.repo_root)


def build_torchrec_command(
    *,
    cfg: BenchmarkConfig,
    memory_mode: str,
    client: ClientSpec,
    run_id: str,
    rdzv_id: str | None = None,
) -> list[str]:
    output_dir = cfg.output_dir / "outputs" / run_id
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={_cuda_visible_devices(client)}",
        *(f"{key}={value}" for key, value in _torchrec_nccl_env().items()),
        cfg.python_bin,
        str(client.repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        "torchrec",
        "--batch-size",
        str(cfg.batch_size),
        "--embedding-dim",
        str(cfg.embedding_dim),
        "--num-embeddings",
        str(cfg.num_embeddings),
        "--steps",
        str(cfg.steps),
        "--warmup-steps",
        str(cfg.warmup_steps),
        "--dense-arch-layer-sizes",
        _dense_arch_for_embedding_dim(cfg.embedding_dim),
        "--data-dir",
        str(cfg.dataset_path),
        "--output-root",
        str(cfg.output_dir),
        "--run-id",
        run_id,
        "--torchrec-memory-mode",
        memory_mode,
        "--torchrec-main-csv",
        str(output_dir / "torchrec_main.csv"),
        "--torchrec-main-agg-csv",
        str(output_dir / "torchrec_main_agg.csv"),
        "--no-start-server",
        "--nnodes",
        str(_client_node_count(cfg.clients)),
        "--node-rank",
        str(client.node_rank),
        "--nproc-per-node",
        str(client.nproc_per_node),
        "--master-addr",
        cfg.clients[0].ip,
        "--master-port",
        str(cfg.master_port),
        "--rdzv-id",
        rdzv_id or run_id,
    ]
    return _wrap_remote(cmd, ssh_host=client.ssh_host, ssh_port=client.ssh_port, cwd=client.repo_root)
