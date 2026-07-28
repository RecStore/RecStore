from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
os.environ.setdefault("RECSTORE_DEFER_OPS_LOAD", "1")
import recstore

from ..config import (
    RunConfig,
    dump_run_config,
    ensure_shared_dir,
    resolve_num_embeddings_per_feature,
    validate_recstore_config,
)
from ..data.dlrm_source import (
    build_kjt_batch_from_dense_sparse_labels,
    build_train_dataloader,
    convert_kjt_ids_to_fused_ids,
    get_default_cat_names,
    inject_project_paths,
)
from ..models.dlrm import (
    build_criterion,
    build_dense_module,
    compute_dense_loss,
    prepare_hybrid_dlrm_input,
    reshape_torchrec_embeddings_for_dlrm,
)
from python.pytorch.recstore.benchmark.report import finalize_recstore_row, summarize_us
from ..runtime.timing import StepTimer
from ..runtime.worker_common import (
    barrier_for_step_alignment as _barrier_for_step_alignment,
    bool_int as _bool_int,
    load_rows as _load_rows,
    parse_nccl_transport_log as _parse_nccl_transport_log,
    pick_socket_ifname as _pick_socket_ifname,
    write_rows as _write_rows,
)
from .base import BenchmarkRunner

from recstore.embedding_read_path import build_embedding_read_path


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator


def _add_sparse_id_stats(
    row: dict[str, Any],
    sparse_features: Any,
    table_offsets: dict[str, int],
    *,
    precomputed: tuple[int, int] | None = None,
) -> None:
    """Record raw/unique/dedup id stats for a batch.

    ``precomputed`` supplies (unique_count, raw_count) when the caller already
    computed them (same-step fused async path), avoiding a second unique() pass.
    """
    if precomputed is not None:
        unique_count, raw_count = int(precomputed[0]), int(precomputed[1])
    elif not hasattr(sparse_features, "keys"):
        raw_count = unique_count = 0
    else:
        fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
        raw_count = int(fused_ids.numel())
        unique_count = int(torch.unique(fused_ids).numel()) if raw_count > 0 else 0
    row["batch_raw_ids"] = raw_count
    row["batch_unique_ids"] = unique_count
    row["batch_dedup_ratio"] = _safe_ratio(raw_count - unique_count, raw_count)


def _finalize_step_timing(
    row: dict[str, Any], *, consume_start: float, wall_start: float
) -> None:
    total_ms = (time.perf_counter() - wall_start) * 1e3
    row["step_total_ms"] = total_ms
    row["step_end_to_end_ms"] = total_ms
    row["samples_per_sec"] = _safe_ratio(row["batch_size"] * 1000.0, total_ms)
    row["batches_per_sec"] = _safe_ratio(1000.0, total_ms)


def _consume_perf_stats(obj: Any) -> dict[str, float]:
    consume = getattr(obj, "consume_perf_stats", None)
    if consume is None:
        return {}
    stats = consume(reset=True)
    return stats if isinstance(stats, dict) else {}


def _merge_consumed_perf_stats(row: dict[str, Any], stats: dict[str, float]) -> None:
    for key, value in stats.items():
        if key in row and row[key] not in (0, 0.0, "", None):
            continue
        row[key] = value


def _reset_perf_stats(obj: Any) -> None:
    reset = getattr(obj, "reset_perf_stats", None)
    if reset is not None:
        reset()


def _maybe_warmup_gpu_local_shm_fast_path(
    cfg: RunConfig, client: Any, device: torch.device
) -> bool:
    if cfg.nnodes != 1 or cfg.single_node_ps_backend != "local_shm":
        return False
    if device.type != "cuda":
        return False
    if not client.is_shared_local_shm_table():
        return False
    # ponytail: activate_shard only on some test/distributed fakes, not RecStoreClient
    activate_shard = getattr(client, "activate_shard", None)
    if callable(activate_shard):
        activate_shard(0)
    if client.current_ps_backend() != "local_shm":
        client.set_ps_backend("local_shm")
    return bool(client.warmup_local_lookup_flat_cuda_region())


def _debug_log_path(cfg: RunConfig, rank: int) -> Path:
    return Path(cfg.output_root) / "outputs" / cfg.run_id / f"recstore_worker_rank{rank}.log"


def _append_worker_debug(cfg: RunConfig, rank: int, message: str) -> None:
    debug_path = _debug_log_path(cfg, rank)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} rank={rank} {message}\n")


def _merge_rank_outputs(paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in paths:
        for row in _load_rows(path):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = ""
                elif key in {"backend", "dist_mode"}:
                    normalized[key] = value
                else:
                    try:
                        normalized[key] = float(value) if "." in value else int(value)
                    except (TypeError, ValueError):
                        normalized[key] = value
            merged.append(normalized)
    merged.sort(key=lambda row: (int(row.get("rank", 0)), int(row.get("step", 0))))
    _write_rows(out_path, merged)
    return merged


def _build_train_dataloader_for_mode(repo_root: Path, cfg: RunConfig, rank: int):
    world_size = cfg.nnodes * cfg.nproc_per_node
    return build_train_dataloader(
        repo_root=repo_root,
        data_dir_rel=cfg.data_dir,
        train_ratio=cfg.train_ratio,
        num_embeddings=cfg.num_embeddings,
        num_embeddings_per_feature=cfg.num_embeddings_per_feature,
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
        rank=rank if world_size > 1 else None,
        world_size=world_size if world_size > 1 else None,
    )


def _maybe_wrap_dense_module_for_dist(
    dense_module: torch.nn.Module, device: torch.device, local_rank: int, use_dist: bool
) -> torch.nn.Module:
    if not use_dist:
        return dense_module
    if device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            dense_module, device_ids=[local_rank], output_device=local_rank
        )
    return torch.nn.parallel.DistributedDataParallel(dense_module)


class RecStoreRunner(BenchmarkRunner):
    """Train a DLRM-style model with embeddings served by the RecStore PS.

    A single run is one of: an in-process single worker, a torchrun launcher
    (re-invokes this module per rank and merges their CSVs), or one distributed
    worker.  Per step it times lookup -> dense fwd/bwd -> dense opt -> sparse
    update. Embedding reads follow ``cfg.read_mode`` (direct / prefetch).
    """

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _rank_output_dir(self, cfg: RunConfig) -> Path:
        return Path(cfg.output_root) / "outputs" / cfg.run_id / "recstore_ranks"

    def _build_torchrun_cmd(self, repo_root: Path, cfg: RunConfig, config_json: Path) -> list[str]:
        # The whole resolved config is handed to each worker as JSON, so the
        # launcher argv stays tiny and never drifts from the config schema.
        return [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", str(cfg.nnodes),
            "--node_rank", str(cfg.node_rank),
            "--nproc_per_node", str(cfg.nproc_per_node),
            "--rdzv_backend", str(cfg.rdzv_backend),
            "--rdzv_endpoint", f"{cfg.master_addr}:{cfg.master_port}",
            "--rdzv_id", str(cfg.rdzv_id),
            "--tee", "3",
            str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
            "--run-config-json", str(config_json),
        ]

    def _run_single_process(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        return self._run_local_worker(
            repo_root=repo_root, cfg=cfg, rank=0, world_size=1, local_rank=0,
            out_csv=Path(cfg.recstore_main_csv),
        )

    def _run_distributed(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rank_dir = self._rank_output_dir(cfg)
        ensure_shared_dir(rank_dir)
        # Workers must not start their own PS server (the launcher already did).
        worker_cfg = dataclasses.replace(cfg, start_server=False)
        config_json = dump_run_config(worker_cfg, rank_dir / "worker_config.json")

        cmd = self._build_torchrun_cmd(repo_root, cfg, config_json)
        env = os.environ.copy()
        env["RS_DEMO_RECSTORE_WORKER"] = "1"
        env["RS_DEMO_RECSTORE_WORKER_DIR"] = str(rank_dir)
        socket_ifname = _pick_socket_ifname()
        if socket_ifname:
            env.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
            env.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
        env.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
        env.setdefault("NCCL_DEBUG", "WARN")
        res = subprocess.run(
            cmd, cwd=str(repo_root), env=env, check=False, text=True, capture_output=True
        )
        if res.returncode != 0:
            raise RuntimeError(
                "recstore torchrun worker failed\n"
                f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )

        world_size = cfg.nnodes * cfg.nproc_per_node
        rank_csvs = [rank_dir / f"rank{rank}.csv" for rank in range(world_size)]
        missing = [str(path) for path in rank_csvs if not path.exists()]
        if missing:
            raise RuntimeError(f"missing rank csv outputs: {missing}")
        rows = _merge_rank_outputs(rank_csvs, Path(cfg.recstore_main_csv))
        return {"backend": "recstore", "rows": rows}

    # -- worker setup ------------------------------------------------------

    def _init_process_group(self, cfg, rank, world_size, local_rank, dist):
        """Set up the CUDA device and (if distributed) the process group."""
        use_dist = world_size > 1
        backend = "nccl"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        _append_worker_debug(
            cfg, rank,
            f"worker_start world_size={world_size} local_rank={local_rank} backend={backend}",
        )
        if use_dist and not dist.is_initialized():
            nccl_log_path = Path(
                os.environ.get(
                    "NCCL_DEBUG_FILE",
                    str(Path(cfg.output_root) / "outputs" / cfg.run_id
                        / f"recstore_nccl_rank{rank}.log"),
                )
            )
            nccl_log_path.parent.mkdir(parents=True, exist_ok=True)
            os.environ["NCCL_DEBUG_FILE"] = str(nccl_log_path)
            os.environ.setdefault("NCCL_DEBUG", "INFO")
            os.environ.setdefault("NCCL_DEBUG_SUBSYS", "NET")
            dist.init_process_group(
                backend=backend,
                device_id=device if device.type == "cuda" else None,
            )
            dist.barrier()
            _append_worker_debug(
                cfg, rank, f"nccl_transport={_parse_nccl_transport_log(nccl_log_path)}"
            )
        return device, use_dist

    def _build_embedding_module(self, cfg, client, default_cat_names):
        """Build the RecStore-backed EmbeddingBagCollection and its table offsets."""
        num_embeddings_per_feature = resolve_num_embeddings_per_feature(
            cfg.num_embeddings, cfg.num_embeddings_per_feature
        )
        eb_configs = [
            {
                "name": f"t_{feature_name}",
                "num_embeddings": int(num_embeddings_per_feature[feature_idx]),
                "embedding_dim": int(cfg.embedding_dim),
                "feature_names": [feature_name],
            }
            for feature_idx, feature_name in enumerate(default_cat_names)
        ]
        if cfg.recstore_enable_fusion:
            table_offsets = {
                cfg_item["feature_names"][0]: feature_idx << cfg.fuse_k
                for feature_idx, cfg_item in enumerate(eb_configs)
            }
        else:
            table_offsets = {cfg_item["feature_names"][0]: 0 for cfg_item in eb_configs}
        return eb_configs, table_offsets, recstore.RecStoreEmbeddingBagCollection

    def _run_local_worker(
        self,
        repo_root: Path,
        cfg: RunConfig,
        rank: int,
        world_size: int,
        local_rank: int,
        out_csv: Path,
    ) -> dict[str, Any]:
        inject_project_paths(repo_root)
        from torch import distributed as dist

        default_cat_names = get_default_cat_names()
        print(f"[rs_demo] repo_root={repo_root}")
        print("[rs_demo] backend=recstore")
        if cfg.read_mode == "direct" and cfg.prefetch_depth > 0:
            print(
                "[rs_demo] read_mode=direct ignores prefetch_depth="
                f"{cfg.prefetch_depth}"
            )

        orig_cwd = Path.cwd()
        try:
            os.chdir(str(self.runtime_dir))
            torch.manual_seed(cfg.seed)
            device, use_dist = self._init_process_group(
                cfg, rank, world_size, local_rank, dist
            )

            recstore.load_ops_library()
            client = recstore.RecStoreClient()
            if cfg.nnodes == 1:
                client.set_ps_backend(cfg.single_node_ps_backend)
            elif cfg.ps_type.upper() == "RDMA":
                client.set_ps_backend("rdma")

            dataset, dataloader = _build_train_dataloader_for_mode(repo_root, cfg, rank)

            eb_configs, table_offsets, RecStoreEmbeddingBagCollection = (
                self._build_embedding_module(cfg, client, default_cat_names)
            )
            fused_id_offsets = torch.tensor(
                [table_offsets[name] for name in default_cat_names],
                dtype=torch.int64,
            )
            embedding_module = RecStoreEmbeddingBagCollection(
                embedding_bag_configs=eb_configs,
                enable_fusion=cfg.recstore_enable_fusion,
                fusion_k=cfg.fuse_k,
                kv_client=client,
                initialize_tables=(rank == 0),
            )
            read_path = build_embedding_read_path(
                cfg.read_mode,
                embedding_module=embedding_module,
                prefetch_depth=cfg.prefetch_depth,
                embedding_dim=cfg.embedding_dim,
                feature_offsets=fused_id_offsets,
            )

            _barrier_for_step_alignment(
                dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
            )

            dense_module = build_dense_module(
                cfg,
                num_sparse_features=len(default_cat_names),
                embedding_dim=cfg.embedding_dim,
                device=device,
            )
            dense_module = _maybe_wrap_dense_module_for_dist(
                dense_module=dense_module, device=device,
                local_rank=local_rank, use_dist=use_dist,
            )
            dispatch_module = (
                dense_module.module
                if isinstance(dense_module, torch.nn.parallel.DistributedDataParallel)
                else dense_module
            )
            criterion = build_criterion(cfg, dispatch_module)
            dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)
            sparse_optimizer = recstore.SparseSGD([embedding_module], lr=0.01)

            if _maybe_warmup_gpu_local_shm_fast_path(cfg=cfg, client=client, device=device):
                print("[rs_demo] warmed local_shm lookup payload region for GPU fast path")
                _barrier_for_step_alignment(
                    dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
                )

            rows: list[dict[str, Any]] = []
            data_iter_state = {"iter": iter(dataloader)}
            prepared_batches: deque = deque()

            def prepare_next_batch(batch_step: int):
                row: dict[str, Any] = {
                    "rank": rank,
                    "batch_size": cfg.batch_size,
                    "step": batch_step,
                    "warmup_excluded": _bool_int(batch_step < cfg.warmup_steps),
                }
                batch_prepare_start = time.perf_counter()
                try:
                    dense_batch, sparse_batch, labels_batch = next(data_iter_state["iter"])
                except StopIteration:
                    data_iter_state["iter"] = iter(dataloader)
                    dense_batch, sparse_batch, labels_batch = next(data_iter_state["iter"])
                row["batch_prepare_ms"] = (time.perf_counter() - batch_prepare_start) * 1e3

                input_pack_start = time.perf_counter()
                _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                    dense_batch, sparse_batch, labels_batch, device=device
                )
                row["input_pack_ms"] = (time.perf_counter() - input_pack_start) * 1e3

                ticket = read_path.on_batch_prepared(
                    batch_step, sparse_features, sparse_batch, row
                )
                if (
                    isinstance(ticket, tuple)
                    and len(ticket) == 3
                    and hasattr(ticket[0], "numel")
                ):
                    unique_ids, _, raw_count = ticket
                    _add_sparse_id_stats(
                        row,
                        sparse_features,
                        table_offsets,
                        precomputed=(int(unique_ids.numel()), int(raw_count)),
                    )
                else:
                    _add_sparse_id_stats(row, sparse_features, table_offsets)
                return (
                    batch_step, row, time.perf_counter(),
                    dense_batch, sparse_features, labels_batch, ticket,
                )

            for step in range(cfg.steps):
                step_wall_start = time.perf_counter()
                observed_depth = read_path.depth * 2
                while (
                    len(prepared_batches) <= observed_depth
                    and step + len(prepared_batches) < cfg.steps
                ):
                    prepared_batches.append(prepare_next_batch(step + len(prepared_batches)))
                if step + len(prepared_batches) >= cfg.steps:
                    read_path.advance_all()

                (
                    _, row, step_start, dense_batch, sparse_features, labels_batch,
                    ticket,
                ) = prepared_batches.popleft()
                consume_step_start = time.perf_counter()
                del step_start  # issue-queue wait is not an E2E export metric

                _reset_perf_stats(embedding_module)
                sparse_optimizer.zero_grad()
                timer = StepTimer(row, torch, device)
                # embed_lookup and sparse_update hit the PS over the network (host
                # + network work) so they stay on the wall clock; the pure-GPU
                # dense stages use CUDA events via timer.gpu().
                with timer.cpu("embed_lookup_ms"):
                    read_path.before_lookup(step, sparse_features, ticket, row)
                    embeddings = embedding_module(sparse_features)

                if embeddings is None:
                    raise RuntimeError("recstore embedding module returned no embeddings")

                with timer.gpu("embed_pool_local_ms"):
                    embedded_sparse_source = reshape_torchrec_embeddings_for_dlrm(
                        embeddings=embeddings, feature_names=default_cat_names, torch=torch
                    )
                with timer.gpu("output_unpack_ms"):
                    dense_features, embedded_sparse, labels = prepare_hybrid_dlrm_input(
                        dense_batch=dense_batch,
                        embedded_sparse_source=embedded_sparse_source,
                        labels_batch=labels_batch,
                        torch=torch, device=device, detach_sparse=True,
                    )
                with timer.gpu("dense_fwd_ms"):
                    loss, _ = compute_dense_loss(
                        cfg, dense_module, criterion, dense_features, embedded_sparse, labels
                    )

                with timer.gpu("backward_ms"):
                    for param in dense_module.parameters():
                        if param.requires_grad:
                            param.grad = None
                    if not embedded_sparse.is_leaf:
                        embedded_sparse.retain_grad()
                    embedded_sparse.grad = None
                    loss.backward()
                    if embedded_sparse.grad is None:
                        raise RuntimeError(
                            "missing embedded_sparse gradient after backward"
                        )
                    embedded_sparse_grad = embedded_sparse.grad.detach()

                with timer.gpu("dense_optimizer_ms"):
                    dense_optimizer.step()
                    dense_optimizer.zero_grad(set_to_none=True)

                with timer.cpu("sparse_optimizer_ms"):
                    replay_start = time.perf_counter()
                    embedded_sparse_source.backward(
                        embedded_sparse_grad.to(embedded_sparse_source.device)
                    )
                    row["sparse_backward_replay_ms"] = (time.perf_counter() - replay_start) * 1e3

                    optimizer_step_start = time.perf_counter()
                    sparse_optimizer.step()
                    row["sparse_optimizer_step_ms"] = (
                        time.perf_counter() - optimizer_step_start
                    ) * 1e3

                    # Overlap PS update latency by preparing future batches.
                    overlap_prepare_start = time.perf_counter()
                    while (
                        len(prepared_batches) <= observed_depth
                        and step + 1 + len(prepared_batches) < cfg.steps
                    ):
                        prepared_batches.append(
                            prepare_next_batch(step + 1 + len(prepared_batches))
                        )
                    row["update_overlap_prepare_ms"] = (
                        time.perf_counter() - overlap_prepare_start
                    ) * 1e3

                    flush_start = time.perf_counter()
                    sparse_optimizer.flush()
                    row["sparse_optimizer_flush_ms"] = (time.perf_counter() - flush_start) * 1e3

                    read_path.after_sparse_update(
                        step, sparse_features, sparse_optimizer, row
                    )
                    sparse_optimizer.zero_grad()

                # Resolve the CUDA-event GPU stages after a single device drain.
                timer.finish()
                row["loss"] = float(loss.detach().float().cpu().item())
                _merge_consumed_perf_stats(row, _consume_perf_stats(embedding_module))

                row["dense_compute_ms"] = (
                    row["dense_fwd_ms"]
                    + row["backward_ms"]
                    + row["dense_optimizer_ms"]
                )
                _barrier_for_step_alignment(
                    dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
                )
                _finalize_step_timing(
                    row, consume_start=consume_step_start, wall_start=step_wall_start
                )

                rows.append(finalize_recstore_row(row))

                if (step + 1) % 10 == 0:
                    print(
                        f"[rs_demo] step {step + 1}/{cfg.steps} "
                        f"emb={rows[-1]['emb_stage_ms']:.2f}ms step={rows[-1]['step_total_ms']:.2f}ms"
                    )

            print("[rs_demo] workload finished")

            _write_rows(out_csv, rows)
            if use_dist and dist.is_initialized():
                dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
                dist.destroy_process_group()
            return {
                "backend": "recstore",
                "rows": rows,
            }
        finally:
            os.chdir(str(orig_cwd))

    def run(self, repo_root: Path, cfg: RunConfig) -> dict:
        if cfg.backend != "recstore":
            raise ValueError("RecStoreRunner requires cfg.backend to be 'recstore'.")
        validate_recstore_config(cfg)

        if os.environ.get("RS_DEMO_RECSTORE_WORKER") == "1":
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(
                os.environ.get("WORLD_SIZE", str(cfg.nnodes * cfg.nproc_per_node))
            )
            worker_dir = Path(os.environ["RS_DEMO_RECSTORE_WORKER_DIR"])
            ensure_shared_dir(worker_dir)
            return self._run_local_worker(
                repo_root=repo_root, cfg=cfg, rank=rank, world_size=world_size,
                local_rank=local_rank, out_csv=worker_dir / f"rank{rank}.csv",
            )

        if cfg.nnodes * cfg.nproc_per_node <= 1:
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
