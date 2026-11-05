from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..config import (
    RunConfig,
    ensure_shared_dir,
    resolve_num_embeddings_per_feature,
    validate_torchrec_config,
)
from ..models.dlrm import (
    build_criterion,
    build_dense_module,
    compute_dense_loss,
)
from ..models.utils import (
    prepare_hybrid_dlrm_input,
    reshape_torchrec_embeddings_for_dlrm,
    run_hybrid_backward,
)
from python.pytorch.recstore.benchmark.report import finalize_torchrec_row
from ..runtime.timing import StepTimer
from ..runtime.continuous_loop import ContinuousLoop, continuous_enabled, verify_continuous_protocol
from ..runtime.worker_common import (
    barrier_for_step_alignment as _barrier_for_step_alignment,
    bool_int as _bool_int,
    read_worker_context as _read_worker_context,
    write_rows as _write_rows,
)
from python.pytorch.recstore.analysis.profiler import ProfilerConfig, build_torchrec_profiler
from .base import BenchmarkRunner, build_rank_dataloader, wrap_dense_module_for_dist


def ensure_torchrec_available() -> None:
    try:
        import torchrec.datasets.criteo  # noqa: F401
        import torchrec.distributed.model_parallel  # noqa: F401
        import torchrec.modules.embedding_configs  # noqa: F401
        import torchrec.modules.embedding_modules  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchRec backend requires the `torchrec` package to be installed."
        ) from exc


def _build_worker_fingerprint(repo_root: Path) -> dict[str, dict[str, str]]:
    rel_paths = [
        "model_zoo/rs_demo/config.py",
        "model_zoo/rs_demo/data/dlrm_source.py",
        "model_zoo/rs_demo/runners/torchrec_runner.py",
        "model_zoo/rs_demo/models/dlrm.py",
        "model_zoo/rs_demo/models/utils.py",
        "model_zoo/rs_demo/runtime/hybrid_dlrm.py",
        "model_zoo/rs_demo/runtime/dense_graph.py",
        "model_zoo/rs_demo/runtime/continuous_loop.py",
    ]
    files: dict[str, str] = {}
    for rel_path in rel_paths:
        path = repo_root / rel_path
        try:
            files[rel_path] = hashlib.md5(path.read_bytes()).hexdigest()
        except FileNotFoundError:
            continue
    return {"files": files}


def _write_or_verify_worker_fingerprint(
    rank: int,
    world_size: int,
    fingerprint: dict[str, dict[str, str]],
    fingerprint_path: Path,
) -> None:
    del world_size
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = fingerprint_path.with_suffix(fingerprint_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if fingerprint_path.exists():
            content = fingerprint_path.read_text(encoding="utf-8")
            all_fingerprints = json.loads(content) if content.strip() else {}
        else:
            all_fingerprints = {}

        all_fingerprints[str(rank)] = fingerprint
        fingerprint_path.write_text(
            json.dumps(all_fingerprints, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        if rank != 0:
            baseline = all_fingerprints.get("0")
            if baseline is not None and baseline != fingerprint:
                raise RuntimeError(
                    f"worker fingerprint mismatch: rank0={baseline} rank{rank}={fingerprint}"
                )


def _compute_or_load_shared_sharding_plan(
    dist,
    rank: int,
    embedding_module,
    sharders,
    planner,
    plan_path: Path,
):
    # Distribute the sharding plan via a broadcast instead of a shared file:
    # without a shared filesystem rank1 cannot read a plan written by rank0.
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        plan = planner.plan(embedding_module, sharders)
        pending_path = plan_path.with_suffix(plan_path.suffix + ".tmp")
        with pending_path.open("wb") as f:
            pickle.dump(plan, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending_path, plan_path)
        payload = [plan]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _remove_stale_distributed_outputs(cfg: RunConfig, rank_dir: Path) -> None:
    run_output_dir = Path(cfg.output_root) / "outputs" / cfg.run_id
    stale_paths = [
        run_output_dir / "torchrec_worker_fingerprints.json",
        run_output_dir / "torchrec_worker_fingerprints.json.lock",
        run_output_dir / "torchrec_plan.pkl",
        run_output_dir / f"torchrec_nccl_rank{cfg.node_rank}.log",
        Path(cfg.torchrec_main_csv),
        Path(cfg.torchrec_main_agg_csv),
        Path(cfg.torchrec_trace_csv),
    ]
    for path in stale_paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    if rank_dir.exists():
        for path in rank_dir.glob("rank*.csv"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _make_trace_handler(cfg: RunConfig, rank: int):
    def _handler(prof) -> None:
        trace_path = Path(cfg.torchrec_trace_dir) / f"rank{rank}.pt.trace.json"
        prof.export_chrome_trace(str(trace_path))

    return _handler


def _is_fair_remote_mode(cfg: RunConfig, world_size: int) -> bool:
    return cfg.torchrec_dist_mode == "fair_remote" and world_size > 1


def _build_uvm_caching_constraints(
    table_names: list[str],
    parameter_constraints_cls,
    embedding_compute_kernel_cls,
) -> dict[str, Any]:
    try:
        fused_uvm_caching = embedding_compute_kernel_cls.FUSED_UVM_CACHING
    except AttributeError as exc:
        raise RuntimeError(
            "TorchRec UVM caching requires EmbeddingComputeKernel.FUSED_UVM_CACHING. "
            "Install a TorchRec/FBGEMM version that supports fused UVM caching."
        ) from exc
    fused_uvm_caching_value = fused_uvm_caching.value
    return {
        table_name: parameter_constraints_cls(compute_kernels=[fused_uvm_caching_value])
        for table_name in table_names
    }


def _zero_embedding_parameters(module, torch) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.zero_()


def _build_train_dataloader_for_mode(repo_root: Path, cfg: RunConfig, rank: int):
    # fair_remote keeps every trainer on the full dataset, unsharded.
    fair_remote_mode = _is_fair_remote_mode(cfg, cfg.nnodes * cfg.nproc_per_node)
    return build_rank_dataloader(
        repo_root,
        cfg,
        rank,
        shuffle=not fair_remote_mode,
        partition=not fair_remote_mode,
    )


def _maybe_wrap_dense_module_for_dist(
    dense_module,
    device,
    local_rank: int,
    use_dist: bool,
    fair_remote_mode: bool,
):
    return wrap_dense_module_for_dist(
        dense_module, device, local_rank, use_dist, enabled=not fair_remote_mode
    )


def _run_single_or_dist_worker(
    repo_root: Path,
    cfg: RunConfig,
    rank: int,
    world_size: int,
    local_rank: int,
    out_csv: Path,
) -> list[dict[str, Any]]:
    import torch
    from torch import distributed as dist
    from torch import nn

    from ..data.dlrm_source import (
        build_kjt_batch_from_dense_sparse_labels,
        inject_project_paths,
    )

    inject_project_paths(repo_root)

    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES
    from torchrec.distributed.model_parallel import (
        DistributedModelParallel,
        get_default_sharders,
    )
    from torchrec.distributed.embedding_types import EmbeddingComputeKernel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner import ParameterConstraints
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection

    overlap_dense = os.environ.get("RECSTORE_DENSE_REDUCE_OVERLAP") == "1"
    if overlap_dense and (not cfg.dense_cuda_graph or cfg.torchrec_memory_mode != "hbm"
                          or cfg.torchrec_dist_mode == "fair_remote"):
        raise ValueError("Dense overlap requires HBM CUDA graph and no fair_remote")
    is_dist = world_size > 1
    use_uvm_caching = cfg.torchrec_memory_mode == "uvm_caching"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if (is_dist or use_uvm_caching) and not dist.is_initialized():
        dist.init_process_group(backend=backend)
        if backend == "nccl" and is_dist:
            dist.barrier()

    verify_continuous_protocol(torch, dist, device, world_size, cfg.steps)
    continuous_mode = continuous_enabled(cfg, device, world_size, lane="torchrec")
    if is_dist:
        fingerprint_path = Path(cfg.output_root) / "outputs" / cfg.run_id / "torchrec_worker_fingerprints.json"
        fingerprint = _build_worker_fingerprint(repo_root)
        _write_or_verify_worker_fingerprint(
            rank=rank,
            world_size=world_size,
            fingerprint=fingerprint,
            fingerprint_path=fingerprint_path,
        )

    fair_remote_mode = _is_fair_remote_mode(cfg, world_size)
    torch.manual_seed(cfg.seed if fair_remote_mode else cfg.seed + rank)

    _dataset, dataloader = _build_train_dataloader_for_mode(
        repo_root=repo_root,
        cfg=cfg,
        rank=rank,
    )
    data_iter = iter(dataloader)
    num_embeddings_per_feature = resolve_num_embeddings_per_feature(
        cfg.num_embeddings,
        cfg.num_embeddings_per_feature,
    )

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=int(cfg.embedding_dim),
            num_embeddings=int(num_embeddings_per_feature[feature_idx]),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    use_dist = world_size > 1
    use_dmp = use_dist or use_uvm_caching
    embedding_init_device = torch.device("meta") if use_dmp else device
    embedding_module = EmbeddingBagCollection(tables=eb_configs, device=embedding_init_device)
    if use_dmp:
        sharders = get_default_sharders()
        constraints = (
            _build_uvm_caching_constraints(
                table_names=[config.name for config in eb_configs],
                parameter_constraints_cls=ParameterConstraints,
                embedding_compute_kernel_cls=EmbeddingComputeKernel,
            )
            if use_uvm_caching
            else None
        )
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size,
                local_world_size=cfg.nproc_per_node,
                compute_device=device.type,
            ),
            constraints=constraints,
        )
        plan = _compute_or_load_shared_sharding_plan(
            dist=dist,
            rank=rank,
            embedding_module=embedding_module,
            sharders=sharders,
            planner=planner,
            plan_path=Path(cfg.output_root) / "outputs" / cfg.run_id / "torchrec_plan.pkl",
        )
        embedding_module = DistributedModelParallel(
            module=embedding_module,
            device=device,
            sharders=sharders,
            plan=plan,
        )
        collective_mode = "measured_distributed" if use_dist else "not_measured_single_process"
        collective_measured = 1 if use_dist else 0
    else:
        embedding_module = embedding_module.to(device)
        collective_mode = "not_measured_single_process"
        collective_measured = 0

    if cfg.torchrec_align_recstore_init:
        _zero_embedding_parameters(embedding_module, torch)
        torch.manual_seed(cfg.seed)

    dense_module = build_dense_module(
        cfg,
        num_sparse_features=len(DEFAULT_CAT_NAMES),
        embedding_dim=cfg.embedding_dim,
        device=device,
    )
    if cfg.dense_cuda_graph:
        if use_dist:
            with torch.no_grad():
                for value in list(dense_module.parameters()) + list(dense_module.buffers()):
                    dist.broadcast(value, src=0)
    else:
        dense_module = _maybe_wrap_dense_module_for_dist(
            dense_module=dense_module,
            device=device,
            local_rank=local_rank,
            use_dist=use_dist,
            fair_remote_mode=fair_remote_mode,
        )

    _dispatch_module = dense_module.module if isinstance(
        dense_module, torch.nn.parallel.DistributedDataParallel) else dense_module
    criterion = build_criterion(cfg, _dispatch_module)
    dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)
    sparse_optimizer = torch.optim.SGD(embedding_module.parameters(), lr=0.01)
    overlap_groups = {"dense": "WORLD" if use_dist else "local",
                      "dense_buffer_group": "None/default"}
    if overlap_dense and use_dist:
        tables = {
            name: sharding.sharding_type
            for module_plan in plan.plan.values()
            for name, sharding in module_plan.items()
        }
        kinds = set(tables.values())
        if not kinds.issubset({"table_wise", "data_parallel"}):
            raise RuntimeError("Dense overlap only supports TW/DP sharding")
        overlap_groups.update(
            sparse=[{"sharding_type": kind, "group": "WORLD"} for kind in sorted(kinds)],
            table_sharding=tables,
            table_counts={kind: sum(value == kind for value in tables.values())
                          for kind in kinds},
        )
    timing_metadata = {
        "timing_protocol": "aligned_step",
        "timing_mode": cfg.torchrec_timing_sync_mode,
        "gpu_stage_timing_semantics": (
            "cuda_event_elapsed" if device.type == "cuda" and cfg.torchrec_timing_sync_mode == "stage"
            else "host_enqueue_wall" if device.type == "cuda" else "cpu_wall"),
    }

    if continuous_mode:
        timing_metadata.update(timing_protocol="continuous_loop_all_steps", timing_mode="none",
                               gpu_stage_timing_semantics="host_enqueue_wall")
    lifecycle = {"rank": rank, **timing_metadata}
    dense_graph = dense_gradient_buffer = None
    if cfg.dense_cuda_graph:
        from ..runtime.dense_graph import capture_dlrm_dense
        dense_graph, dense_gradient_buffer = capture_dlrm_dense(
            dense_module, criterion, dataloader, cfg.batch_size,
            num_features=len(DEFAULT_CAT_NAMES), embedding_dim=cfg.embedding_dim,
        )
        lifecycle = {
            **timing_metadata,
            "rank": rank, "dense_graph_capture_ms": dense_graph.capture_ms,
            "dense_graph_batch_sizes": list(dense_graph.batch_sizes),
            "dense_graph_memory": dense_graph.memory_stats,
            "dense_graph_scope": "dense forward/backward; individual stage times unavailable",
        }
        lifecycle.update(dense_reduce_overlap_enabled=int(overlap_dense),
            dense_reduce_overlap_groups=overlap_groups,
            gradient_communication_semantics="remaining_join" if overlap_dense else "full_reduce")
        lifecycle_path = Path(cfg.output_root) / "outputs" / cfg.run_id / f"lifecycle_rank{rank}.json"
        lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
        lifecycle_path.write_text(json.dumps(lifecycle, indent=2) + "\n")
        _barrier_for_step_alignment(
            dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
        )

    profiler = build_torchrec_profiler(
        ProfilerConfig(
            enabled=cfg.torchrec_profiler,
            trace_dir=cfg.torchrec_trace_dir,
            warmup=cfg.torchrec_profiler_warmup,
            active=cfg.torchrec_profiler_active,
            repeat=cfg.torchrec_profiler_repeat,
        ),
        on_trace_ready=_make_trace_handler(cfg, rank) if cfg.torchrec_profiler else None,
    )
    profiler_context = profiler or nullcontext()

    def prepare_batch():
        nonlocal data_iter
        start = time.perf_counter()
        try:
            dense, sparse, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            dense, sparse, labels = next(data_iter)
        prepare_ms = (time.perf_counter() - start) * 1e3
        start = time.perf_counter()
        dense, features = build_kjt_batch_from_dense_sparse_labels(dense, sparse, labels)
        features = features.to(device, non_blocking=True)
        pack_ms = (time.perf_counter() - start) * 1e3
        return dense, features, labels, prepare_ms, pack_ms

    prepared_batch = None
    rows: list[dict[str, Any]] = []
    is_trainer_rank = (not fair_remote_mode) or rank == 0
    continuous = ContinuousLoop(torch, device, cfg.steps) if continuous_mode else None
    with profiler_context, continuous or nullcontext():
        for step in range(cfg.steps):
            step_start = time.perf_counter() if continuous is not None else None
            backpressure_ms = continuous.admit(step) if continuous is not None else 0.0
            row: dict[str, Any] = {
                **timing_metadata,
                "backend": "torchrec",
                "nproc": world_size,
                "rank": rank,
                "batch_size": cfg.batch_size,
                "step": step,
                "warmup_excluded": _bool_int(step < cfg.warmup_steps),
                "collective_mode": collective_mode,
                "collective_measured": collective_measured,
                "nnodes": cfg.nnodes,
                "nproc_per_node": cfg.nproc_per_node,
                "world_size": cfg.nnodes * cfg.nproc_per_node,
                "dist_mode": "multi_node" if cfg.nnodes > 1 else "single_node",
                "torchrec_dist_mode": cfg.torchrec_dist_mode,
                "torchrec_memory_mode": cfg.torchrec_memory_mode,
                "torchrec_timing_sync_mode": "none" if continuous_mode else cfg.torchrec_timing_sync_mode,
                "torchrec_role": "trainer" if is_trainer_rank else "embedding_worker",
                "torchrec_is_trainer": _bool_int(is_trainer_rank),
            }
            if continuous is None:
                step_start = time.perf_counter()
            timer = StepTimer(row, torch, device, mode="none" if continuous_mode else cfg.torchrec_timing_sync_mode)
            dense_work = None
            row["dense_reduce_overlap_enabled"] = int(overlap_dense)
            row["gradient_communication_enqueue_ms"] = 0.0
            row["gradient_communication_semantics"] = "remaining_join" if overlap_dense else "full_reduce"

            if prepared_batch is None:
                prepared_batch = prepare_batch()
            dense_batch, sparse_features, labels_batch, prepare_ms, pack_ms = prepared_batch
            prepared_batch = None
            row["batch_prepare_ms"] = prepare_ms
            row["input_pack_ms"] = pack_ms
            row["batch_size"] = int(dense_batch.shape[0])

            with timer.gpu("embed_lookup_ms"):
                embeddings = embedding_module(sparse_features)

            with timer.gpu("embed_pool_local_ms"):
                embedded_sparse_source = reshape_torchrec_embeddings_for_dlrm(
                    embeddings=embeddings,
                    feature_names=DEFAULT_CAT_NAMES,
                    torch=torch,
                )

            with timer.gpu("output_unpack_ms"):
                dense_features, embedded_sparse, labels = prepare_hybrid_dlrm_input(
                    dense_batch=dense_batch,
                    embedded_sparse_source=embedded_sparse_source,
                    labels_batch=labels_batch,
                    torch=torch,
                    device=device,
                    detach_sparse=True,
                    synchronize=not cfg.dense_cuda_graph,
                )

            if is_trainer_rank:
                if dense_graph is not None:
                    with timer.gpu("dense_graph_ms"):
                        loss, embedded_sparse_grad = dense_graph(
                            dense_features, embedded_sparse, labels
                        )
                    row["dense_fwd_ms"] = row["backward_ms"] = ""
                    if overlap_dense:
                        with timer.cpu("gradient_communication_enqueue_ms"):
                            dense_work = dense_gradient_buffer.submit_reduce()
                    else:
                        with timer.gpu("gradient_communication_ms"):
                            dense_gradient_buffer.reduce()
                else:
                    with timer.gpu("dense_fwd_ms"):
                        loss, _ = compute_dense_loss(
                            cfg, dense_module, criterion,
                            dense_features, embedded_sparse, labels)
                    with timer.gpu("backward_ms"):
                        embedded_sparse_grad = run_hybrid_backward(
                            loss=loss,
                            embedded_sparse=embedded_sparse,
                            dense_module=dense_module,
                            torch=torch,
                            device=device,
                            synchronize=False,
                        )

                if not overlap_dense:
                    with timer.gpu("dense_optimizer_ms"):
                        dense_optimizer.step()
                        dense_optimizer.zero_grad(set_to_none=True)
            else:
                row["dense_fwd_ms"] = 0.0
                row["backward_ms"] = 0.0
                row["dense_optimizer_ms"] = 0.0
                embedded_sparse_grad = torch.zeros_like(embedded_sparse)

            try:
                embedded_sparse_grad = embedded_sparse_grad.contiguous()
                with timer.gpu("sparse_optimizer_ms"):
                    if fair_remote_mode and use_dist:
                        dist.broadcast(embedded_sparse_grad, src=0)
                    # Preserve TorchRec's internal TW/DP collective order.
                    embedded_sparse_source.backward(
                        embedded_sparse_grad.to(embedded_sparse_source.device)
                    )
                    sparse_optimizer.step()
                    sparse_optimizer.zero_grad(set_to_none=True)
            except BaseException as sparse_error:
                from ..runtime.dense_graph import wait_dense_after_error
                wait_dense_after_error(dense_work, sparse_error)
                raise
            if overlap_dense:
                # The detached embedding backward does not read dense parameters.
                with timer.gpu("gradient_communication_ms"):
                    if dense_work is not None:
                        dense_work.wait()
                with timer.gpu("dense_optimizer_ms"):
                    dense_optimizer.step()
                    dense_optimizer.zero_grad(set_to_none=True)

            if cfg.dense_cuda_graph and step + 1 < cfg.steps:
                # Match the PS pipeline's one-batch CPU lookahead. This only
                # prepares inputs; next-step embedding reads stay after SGD.
                with timer.cpu("update_overlap_prepare_ms"):
                    prepared_batch = prepare_batch()

            if is_trainer_rank:
                # Match RecStore's readback boundary; stream dependencies and
                # the final device drain already order the backward/update work.
                if continuous is not None:
                    continuous.record_loss_and_terminal(step, loss)
                    row["loss"] = ""
                else:
                    row["loss"] = float(loss.detach().float().cpu().item())
            if profiler is not None:
                profiler.step()

            # GPU stages are timed with CUDA events and resolved in finish() after
            # a single device drain, so no stage absorbs a neighbor's un-drained
            # tail. finish() returns that drain wait (the cross-rank straggler
            # cost) instead of letting it vanish into the step_total gap.
            row["step_sync_wait_ms"] = timer.finish()
            row["dense_compute_ms"] = row["dense_optimizer_ms"] + (
                row["dense_graph_ms"] if dense_graph is not None
                else row["dense_fwd_ms"] + row["backward_ms"]
            )
            if continuous is None:
                row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
            else:
                continuous.finish_row(row, step_start, backpressure_ms)
            row["collective_launch_ms"] = 0.0
            row["collective_wait_ms"] = (
                row["embed_lookup_ms"] if use_dist else 0.0
            )
            rows.append(finalize_torchrec_row(row))
            if continuous is None:
                _barrier_for_step_alignment(
                    dist=dist,
                    device=device,
                    local_rank=local_rank,
                    use_dist=use_dist,
                )
                rows[-1]["step_wall_ms"] = (time.perf_counter() - step_start) * 1e3
            else:
                rows[-1]["host_step_enqueue_wall_ms"] = (time.perf_counter() - step_start) * 1e3
        if continuous is not None:
            loop_summary = continuous.finish(rows, lambda: _barrier_for_step_alignment(
                dist=dist, device=device, local_rank=local_rank, use_dist=use_dist))
            loop_summary.update(rank=rank, backend="torchrec", world_size=world_size,
                                configured_warmup_steps=cfg.warmup_steps, run_id=cfg.run_id)
            lifecycle["continuous_loop"] = loop_summary
            loop_path = Path(cfg.output_root) / "outputs" / cfg.run_id / f"continuous_rank{rank}.json"
            loop_path.parent.mkdir(parents=True, exist_ok=True)
            loop_path.write_text(json.dumps(loop_summary, indent=2) + "\n")

    if device.type == "cuda":
        lifecycle["training_cuda_memory"] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    lifecycle_path = Path(cfg.output_root) / "outputs" / cfg.run_id / f"lifecycle_rank{rank}.json"
    lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
    lifecycle_path.write_text(json.dumps(lifecycle, indent=2) + "\n")
    _write_rows(out_csv, rows)
    if cfg.validation_state:
        from ..runtime.validation import export_validation_state
        export_validation_state(
            Path(cfg.output_root) / "outputs" / cfg.run_id / f"validation_rank{rank}.pt",
            embedding_module, dense_module,
        )
    if (is_dist or use_uvm_caching) and dist.is_initialized():
        if is_dist:
            dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
        dist.destroy_process_group()
    return rows


class TorchRecRunner(BenchmarkRunner):
    backend = "torchrec"
    rank_dir_name = "torchrec_ranks"

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _main_csv(self, cfg: RunConfig) -> Path:
        return Path(cfg.torchrec_main_csv)

    def _prepare_distributed(self, cfg: RunConfig, rank_dir: Path) -> None:
        _remove_stale_distributed_outputs(cfg, rank_dir)

    def _after_torchrun(self, cfg: RunConfig, rank_dir: Path, result) -> None:
        nccl_log = (
            Path(cfg.output_root)
            / "outputs"
            / cfg.run_id
            / f"torchrec_nccl_rank{cfg.node_rank}.log"
        )
        nccl_log.write_text(result.stdout + result.stderr, encoding="utf-8")

    def _run_single_process(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rows = _run_single_or_dist_worker(
            repo_root=repo_root,
            cfg=cfg,
            rank=0,
            world_size=1,
            local_rank=0,
            out_csv=Path(cfg.torchrec_main_csv),
        )
        return {"backend": "torchrec", "rows": rows}

    def run(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        if cfg.backend != "torchrec":
            raise ValueError("TorchRecRunner requires cfg.backend to be 'torchrec'.")
        validate_torchrec_config(cfg)
        if cfg.steps <= 0:
            raise ValueError("TorchRec runner requires --steps to be greater than 0.")

        ensure_torchrec_available()

        worker = _read_worker_context(
            "torchrec", default_world_size=cfg.nnodes * cfg.nproc_per_node
        )
        if worker is not None:
            ensure_shared_dir(worker.output_dir)
            out_csv = worker.output_dir / f"rank{worker.rank}.csv"
            rows = _run_single_or_dist_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=worker.rank,
                world_size=worker.world_size,
                local_rank=worker.local_rank,
                out_csv=out_csv,
            )
            return {"backend": "torchrec", "rows": rows}

        if cfg.nnodes * cfg.nproc_per_node <= 1 and cfg.torchrec_memory_mode == "hbm":
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
