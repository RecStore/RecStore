from __future__ import annotations

from .common import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_NUM_EMBEDDINGS,
    DEFAULT_REMOTE_SERVER_HOST,
    DEFAULT_STEPS,
    DEFAULT_WARMUP_STEPS,
    LaneSpec,
    MetricSpec,
    SuiteSpec,
)


MAIN_METRICS = (
    MetricSpec("step_total_ms", "main_csv", "step_total_ms"),
    MetricSpec("dense_fwd_ms", "main_csv", "dense_fwd_ms"),
    MetricSpec("backward_ms", "main_csv", "backward_ms"),
    MetricSpec("dense_optimizer_ms", "main_csv", "dense_optimizer_ms"),
)

EMBED_METRICS = (
    MetricSpec("input_pack_ms", "main_csv", "input_pack_ms"),
    MetricSpec("embed_lookup_ms", "main_csv", "embed_lookup_ms"),
    MetricSpec("embed_pool_local_ms", "main_csv", "embed_pool_local_ms"),
    MetricSpec("lookup_local_lookup_ms", "main_csv", "lookup_local_lookup_ms"),
    MetricSpec("lookup_wait_ms", "main_csv", "lookup_wait_ms"),
    MetricSpec("lookup_total_ms", "main_csv", "lookup_total_ms"),
    MetricSpec("lookup_owner_exchange_ms", "main_csv", "lookup_owner_exchange_ms"),
    MetricSpec("lookup_reassemble_ms", "main_csv", "lookup_reassemble_ms"),
    MetricSpec("collective_wait_ms", "main_csv", "collective_wait_ms"),
    MetricSpec("output_unpack_ms", "main_csv", "output_unpack_ms"),
    MetricSpec("emb_stage_ms", "main_csv", "emb_stage_ms"),
    MetricSpec("sparse_optimizer_ms", "main_csv", "sparse_optimizer_ms"),
    MetricSpec("update_owner_exchange_ms", "main_csv", "update_owner_exchange_ms"),
    MetricSpec("update_trace_merge_ms", "main_csv", "update_trace_merge_ms"),
    MetricSpec("update_local_apply_ms", "main_csv", "update_local_apply_ms"),
    MetricSpec("dense_fwd_ms", "main_csv", "dense_fwd_ms"),
    MetricSpec("backward_ms", "main_csv", "backward_ms"),
    MetricSpec("dense_optimizer_ms", "main_csv", "dense_optimizer_ms"),
    MetricSpec("step_total_ms", "main_csv", "step_total_ms"),
)

CHAIN_LOCAL_METRICS = (
    MetricSpec("op_total_us", "chain_csv", "op_total_us"),
    MetricSpec("client_total_us", "chain_csv", "client_total_us"),
    MetricSpec("client_serialize_us", "chain_csv", "client_serialize_us"),
    MetricSpec("client_rpc_us", "chain_csv", "client_rpc_us"),
    MetricSpec("server_total_us", "chain_csv", "server_total_us"),
    MetricSpec("server_backend_update_us", "chain_csv", "server_backend_update_us"),
)

CHAIN_REMOTE_METRICS = (
    MetricSpec("op_total_us", "chain_csv", "op_total_us"),
    MetricSpec("client_total_us", "chain_csv", "client_total_us"),
    MetricSpec("client_serialize_us", "chain_csv", "client_serialize_us"),
    MetricSpec("client_rpc_us", "chain_csv", "client_rpc_us"),
)


def _merge_metrics(*groups: tuple[MetricSpec, ...]) -> tuple[MetricSpec, ...]:
    merged: list[MetricSpec] = []
    seen: set[str] = set()
    for group in groups:
        for metric in group:
            if metric.name in seen:
                continue
            merged.append(metric)
            seen.add(metric.name)
    return tuple(merged)


def _base_lane(name: str, slug: str, backend: str, batch_size: int, **kwargs: object) -> LaneSpec:
    return LaneSpec(
        name=name,
        slug=slug,
        backend=backend,
        steps=DEFAULT_STEPS,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        batch_size=batch_size,
        num_embeddings=DEFAULT_NUM_EMBEDDINGS,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        **kwargs,
    )


def build_suites() -> dict[str, SuiteSpec]:
    recstore_local = _base_lane(
        "RecStore-本地BRPC",
        "recstore-local-brpc",
        "recstore",
        512,
        metrics=_merge_metrics(MAIN_METRICS, EMBED_METRICS, CHAIN_LOCAL_METRICS),
    )
    recstore_local_2proc = _base_lane(
        "RecStore-单机双进程",
        "recstore-local-2proc",
        "recstore",
        256,
        nnodes=1,
        nproc_per_node=2,
        master_port=29672,
        no_start_server=False,
        metrics=MAIN_METRICS,
    )
    recstore_remote = _base_lane(
        "RecStore-远端参数服务器",
        "recstore-remote-ps",
        "recstore",
        512,
        no_start_server=True,
        server_host=DEFAULT_REMOTE_SERVER_HOST,
        server_port0=15123,
        server_port1=15124,
        recstore_runtime_dir="remote_ps",
        needs_remote_ps=True,
        needs_remote_sync=True,
        metrics=_merge_metrics(MAIN_METRICS, EMBED_METRICS, CHAIN_REMOTE_METRICS),
    )
    recstore_2node = _base_lane(
        "RecStore-双机双进程",
        "recstore-2node-2proc",
        "recstore",
        256,
        nnodes=2,
        nproc_per_node=1,
        master_port=29673,
        no_start_server=True,
        server_host=DEFAULT_REMOTE_SERVER_HOST,
        server_port0=15123,
        server_port1=15124,
        recstore_runtime_dir="two_node",
        needs_remote_ps=True,
        needs_remote_sync=True,
        remote_worker=True,
        metrics=_merge_metrics(MAIN_METRICS, EMBED_METRICS, CHAIN_REMOTE_METRICS),
    )
    torchrec_local = _base_lane(
        "TorchRec-单机单卡",
        "torchrec-local-1gpu",
        "torchrec",
        512,
        no_start_server=True,
        metrics=_merge_metrics(MAIN_METRICS, EMBED_METRICS),
    )
    torchrec_local_2proc = _base_lane(
        "TorchRec-单机双进程",
        "torchrec-local-2proc",
        "torchrec",
        256,
        nnodes=1,
        nproc_per_node=2,
        master_port=29674,
        no_start_server=True,
        metrics=_merge_metrics(MAIN_METRICS, EMBED_METRICS),
    )
    torchrec_2node = _base_lane(
        "TorchRec-双机双进程",
        "torchrec-2node-2proc",
        "torchrec",
        256,
        nnodes=2,
        nproc_per_node=1,
        master_port=29675,
        no_start_server=True,
        remote_worker=True,
        needs_remote_sync=True,
        metrics=_merge_metrics(MAIN_METRICS, EMBED_METRICS),
    )
    torchrec_fair = _base_lane(
        "TorchRec-双机单训练端",
        "torchrec-2node-fair",
        "torchrec",
        512,
        nnodes=2,
        nproc_per_node=1,
        master_port=29676,
        no_start_server=True,
        remote_worker=True,
        needs_remote_sync=True,
        torchrec_dist_mode="fair_remote",
        metrics=(MetricSpec("step_total_ms", "main_csv", "step_total_ms"), MetricSpec("collective_wait_ms", "main_csv", "collective_wait_ms")),
    )

    return {
        "main_results": SuiteSpec(
            name="main_results",
            lanes=(
                recstore_local,
                recstore_local_2proc,
                recstore_remote,
                recstore_2node,
                torchrec_local,
                torchrec_local_2proc,
                torchrec_2node,
                torchrec_fair,
            ),
        ),
        "stage_breakdown": SuiteSpec(
            name="stage_breakdown",
            lanes=(
                recstore_local,
                recstore_remote,
                recstore_2node,
                torchrec_local,
                torchrec_local_2proc,
                torchrec_2node,
                torchrec_fair,
            ),
        ),
        "recstore_chain": SuiteSpec(
            name="recstore_chain",
            lanes=(recstore_local, recstore_remote, recstore_2node),
        ),
    }
