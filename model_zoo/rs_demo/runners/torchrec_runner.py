from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..config import (
    RunConfig,
    dump_run_config,
    ensure_shared_dir,
    resolve_num_embeddings_per_feature,
    validate_torchrec_config,
)
from ..models.dlrm import (
    build_criterion,
    build_dense_module,
    compute_dense_loss,
    parse_layer_sizes,
    prepare_hybrid_dlrm_input,
    reshape_torchrec_embeddings_for_dlrm,
    run_hybrid_backward,
)
from python.pytorch.recstore.benchmark.report import finalize_torchrec_row
from ..runtime.timing import StepTimer
from ..runtime.worker_common import (
    barrier_for_step_alignment as _barrier_for_step_alignment,
    bool_int as _bool_int,
    load_rows as _load_rows,
    parse_nccl_transport_log as _parse_nccl_transport_log,
    pick_socket_ifname as _pick_socket_ifname,
    write_rows as _write_rows,
)
from python.pytorch.recstore.analysis.profiler import build_torchrec_profiler
from .base import BenchmarkRunner


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


def _debug_log_path(cfg: RunConfig, rank: int) -> Path:
    return Path(cfg.output_root) / "outputs" / cfg.run_id / f"torchrec_worker_rank{rank}.log"


def _append_worker_debug(cfg: RunConfig, rank: int, message: str) -> None:
    debug_path = _debug_log_path(cfg, rank)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} rank={rank} {message}\n")


def _build_worker_fingerprint(repo_root: Path) -> dict[str, dict[str, str]]:
    rel_paths = [
        "model_zoo/rs_demo/config.py",
        "model_zoo/rs_demo/data/dlrm_source.py",
        "model_zoo/rs_demo/runners/torchrec_runner.py",
        "model_zoo/rs_demo/runtime/hybrid_dlrm.py",
    ]
    files: dict[str, str] = {}
    for rel_path in rel_paths:
        path = repo_root / rel_path
        files[rel_path] = hashlib.md5(path.read_bytes()).hexdigest()
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


def _summarize_sharding_plan(plan: Any) -> str:
    plan_map = getattr(plan, "plan", {})
    if not isinstance(plan_map, dict):
        return f"plan_type={type(plan).__name__}"

    module_summaries: list[str] = []
    for module_path, module_plan in sorted(plan_map.items(), key=lambda item: str(item[0])):
        table_summaries: list[str] = []
        if isinstance(module_plan, dict):
            for table_name, parameter_sharding in sorted(
                module_plan.items(), key=lambda item: str(item[0])
            ):
                sharding_type = getattr(parameter_sharding, "sharding_type", "unknown")
                compute_kernel = getattr(parameter_sharding, "compute_kernel", "unknown")
                ranks = getattr(parameter_sharding, "ranks", None)
                table_summaries.append(
                    f"{table_name}:{sharding_type}:{compute_kernel}:ranks={ranks}"
                )
        module_label = module_path or "<root>"
        module_summaries.append(
            f"module={module_label}[{'; '.join(table_summaries) if table_summaries else 'empty'}]"
        )
    return " | ".join(module_summaries) if module_summaries else "plan=empty"


def _compute_or_load_shared_sharding_plan(
    dist,
    rank: int,
    embedding_module,
    sharders,
    planner,
    plan_path: Path,
):
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        plan = planner.plan(embedding_module, sharders)
        pending_path = plan_path.with_suffix(plan_path.suffix + ".tmp")
        with pending_path.open("wb") as f:
            pickle.dump(plan, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending_path, plan_path)
    else:
        wait_deadline = time.monotonic() + 60.0
        while not plan_path.exists():
            if time.monotonic() >= wait_deadline:
                raise TimeoutError(f"Timed out waiting for shared sharding plan: {plan_path}")
            time.sleep(0.1)
    with plan_path.open("rb") as f:
        plan = pickle.load(f)
    return plan


def _remove_stale_distributed_outputs(cfg: RunConfig, rank_dir: Path) -> None:
    run_output_dir = Path(cfg.output_root) / "outputs" / cfg.run_id
    stale_paths = [
        run_output_dir / "torchrec_worker_fingerprints.json",
        run_output_dir / "torchrec_worker_fingerprints.json.lock",
        run_output_dir / "torchrec_plan.pkl",
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


def _merge_rank_outputs(paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in paths:
        for row in _load_rows(path):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = ""
                    continue
                if key in {"backend", "collective_mode"}:
                    normalized[key] = value
                    continue
                try:
                    if "." in value:
                        normalized[key] = float(value)
                    else:
                        normalized[key] = int(value)
                except (TypeError, ValueError):
                    normalized[key] = value
            merged.append(normalized)
    if any(str(row.get("torchrec_dist_mode", "")) == "fair_remote" for row in merged):
        merged = [row for row in merged if int(row.get("torchrec_is_trainer", 1)) == 1]
    merged.sort(key=lambda row: (int(row.get("rank", 0)), int(row.get("step", 0))))
    _write_rows(out_path, merged)
    return merged


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
    fused_uvm_caching_value = getattr(fused_uvm_caching, "value", fused_uvm_caching)
    return {
        table_name: parameter_constraints_cls(compute_kernels=[fused_uvm_caching_value])
        for table_name in table_names
    }


def _zero_embedding_parameters(module, torch) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.zero_()


def _build_train_dataloader_for_mode(
    repo_root: Path,
    cfg: RunConfig,
    rank: int,
    torch,
):
    from ..data.dlrm_source import build_train_dataloader

    world_size = cfg.nnodes * cfg.nproc_per_node
    fair_remote_mode = _is_fair_remote_mode(cfg, world_size)
    shuffle = not fair_remote_mode
    seed = cfg.seed
    return build_train_dataloader(
        repo_root=repo_root,
        data_dir_rel=cfg.data_dir,
        train_ratio=cfg.train_ratio,
        num_embeddings=cfg.num_embeddings,
        num_embeddings_per_feature=cfg.num_embeddings_per_feature,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        seed=seed,
        rank=rank if (world_size > 1 and not fair_remote_mode) else None,
        world_size=world_size if (world_size > 1 and not fair_remote_mode) else None,
    )


def _maybe_wrap_dense_module_for_dist(
    dense_module,
    device,
    local_rank: int,
    use_dist: bool,
    fair_remote_mode: bool,
    torch,
):
    if not use_dist or fair_remote_mode:
        return dense_module
    if device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            dense_module,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    return torch.nn.parallel.DistributedDataParallel(dense_module)


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

    is_dist = world_size > 1
    use_uvm_caching = cfg.torchrec_memory_mode == "uvm_caching"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    _append_worker_debug(
        cfg,
        rank,
        f"worker_start world_size={world_size} local_rank={local_rank} backend={backend}",
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if (is_dist or use_uvm_caching) and not dist.is_initialized():
        nccl_log_path = None
        if backend == "nccl" and is_dist:
            if "NCCL_DEBUG_FILE" in os.environ:
                nccl_log_path = Path(os.environ["NCCL_DEBUG_FILE"])
            else:
                nccl_log_path = (
                    Path(cfg.output_root)
                    / "outputs"
                    / cfg.run_id
                    / f"torchrec_nccl_rank{rank}.log"
                )
                nccl_log_path.parent.mkdir(parents=True, exist_ok=True)
                os.environ["NCCL_DEBUG_FILE"] = str(nccl_log_path)
            os.environ.setdefault("NCCL_DEBUG", "INFO")
            os.environ.setdefault("NCCL_DEBUG_SUBSYS", "NET")
        _append_worker_debug(cfg, rank, f"before_init_process_group device={device}")
        dist.init_process_group(backend=backend)
        if backend == "nccl" and is_dist:
            dist.barrier()
            _append_worker_debug(
                cfg,
                rank,
                f"nccl_transport={_parse_nccl_transport_log(nccl_log_path)}",
            )
        _append_worker_debug(cfg, rank, "after_init_process_group")

    if is_dist:
        fingerprint_path = Path(cfg.output_root) / "outputs" / cfg.run_id / "torchrec_worker_fingerprints.json"
        fingerprint = _build_worker_fingerprint(repo_root)
        _write_or_verify_worker_fingerprint(
            rank=rank,
            world_size=world_size,
            fingerprint=fingerprint,
            fingerprint_path=fingerprint_path,
        )
        _append_worker_debug(cfg, rank, f"worker_fingerprint {fingerprint}")

    fair_remote_mode = _is_fair_remote_mode(cfg, world_size)
    torch.manual_seed(cfg.seed if fair_remote_mode else cfg.seed + rank)

    _dataset, dataloader = _build_train_dataloader_for_mode(
        repo_root=repo_root,
        cfg=cfg,
        rank=rank,
        torch=torch,
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
    if use_dist:
        _append_worker_debug(cfg, rank, "before_sharding_plan")
    if use_uvm_caching:
        _append_worker_debug(cfg, rank, "torchrec_memory_mode=uvm_caching")
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
        if use_dist:
            _append_worker_debug(cfg, rank, "after_sharding_plan")
        _append_worker_debug(cfg, rank, f"plan_summary {_summarize_sharding_plan(plan)}")
        _append_worker_debug(
            cfg,
            rank,
            f"before_distributed_model_parallel state_dict_keys={list(embedding_module.state_dict().keys())}",
        )
        try:
            embedding_module = DistributedModelParallel(
                module=embedding_module,
                device=device,
                sharders=sharders,
                plan=plan,
            )
        except Exception as exc:
            _append_worker_debug(
                cfg,
                rank,
                f"dmp_init_exception type={type(exc).__name__} message={exc}",
            )
            raise
        _append_worker_debug(cfg, rank, "after_distributed_model_parallel")
        collective_mode = "measured_distributed" if use_dist else "not_measured_single_process"
        collective_measured = 1 if use_dist else 0
    else:
        embedding_module = embedding_module.to(device)
        collective_mode = "not_measured_single_process"
        collective_measured = 0

    if cfg.torchrec_align_recstore_init:
        _zero_embedding_parameters(embedding_module, torch)
        torch.manual_seed(cfg.seed)
        _append_worker_debug(cfg, rank, "torchrec_align_recstore_init=1")

    dense_module = build_dense_module(
        cfg,
        num_sparse_features=len(DEFAULT_CAT_NAMES),
        embedding_dim=cfg.embedding_dim,
        device=device,
    )
    dense_module = _maybe_wrap_dense_module_for_dist(
        dense_module=dense_module,
        device=device,
        local_rank=local_rank,
        use_dist=use_dist,
        fair_remote_mode=fair_remote_mode,
        torch=torch,
    )
    if use_dist and fair_remote_mode:
        _append_worker_debug(cfg, rank, "skip_dense_ddp_fair_remote")

    _dispatch_module = dense_module.module if isinstance(
        dense_module, torch.nn.parallel.DistributedDataParallel) else dense_module
    criterion = build_criterion(cfg, _dispatch_module)
    _append_worker_debug(cfg, rank, "after_criterion")
    _append_worker_debug(cfg, rank, "before_optimizer_init")
    dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)
    sparse_optimizer = torch.optim.SGD(embedding_module.parameters(), lr=0.01)
    _append_worker_debug(cfg, rank, "after_optimizer_init")

    profiler = build_torchrec_profiler(
        cfg,
        on_trace_ready=_make_trace_handler(cfg, rank) if cfg.torchrec_profiler else None,
    )
    profiler_context = profiler or nullcontext()
    _append_worker_debug(cfg, rank, "before_training_loop")

    rows: list[dict[str, Any]] = []
    is_trainer_rank = (not fair_remote_mode) or rank == 0
    with profiler_context:
        for step in range(cfg.steps):
            _append_worker_debug(cfg, rank, f"step_start step={step}")
            row: dict[str, Any] = {
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
                "torchrec_timing_sync_mode": cfg.torchrec_timing_sync_mode,
                "torchrec_role": "trainer" if is_trainer_rank else "embedding_worker",
                "torchrec_is_trainer": _bool_int(is_trainer_rank),
            }
            step_start = time.perf_counter()
            timer = StepTimer(row, torch, device)

            _append_worker_debug(cfg, rank, f"before_batch_prepare step={step}")
            with timer.cpu("batch_prepare_ms"):
                try:
                    dense_batch, sparse_batch, labels_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    dense_batch, sparse_batch, labels_batch = next(data_iter)

            _append_worker_debug(cfg, rank, f"before_input_pack step={step}")
            with timer.cpu("input_pack_ms"):
                dense_batch, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                    dense_batch,
                    sparse_batch,
                    labels_batch,
                )
                sparse_features = sparse_features.to(device, non_blocking=True)

            _append_worker_debug(cfg, rank, f"before_embedding step={step}")
            with timer.gpu("embed_lookup_local_ms"):
                embeddings = embedding_module(sparse_features)

            _append_worker_debug(cfg, rank, f"before_pool step={step}")
            with timer.gpu("embed_pool_local_ms"):
                embedded_sparse_source = reshape_torchrec_embeddings_for_dlrm(
                    embeddings=embeddings,
                    feature_names=DEFAULT_CAT_NAMES,
                    torch=torch,
                )

            _append_worker_debug(cfg, rank, f"before_output_unpack step={step}")
            with timer.gpu("output_unpack_ms"):
                dense_features, embedded_sparse, labels = prepare_hybrid_dlrm_input(
                    dense_batch=dense_batch,
                    embedded_sparse_source=embedded_sparse_source,
                    labels_batch=labels_batch,
                    torch=torch,
                    device=device,
                    detach_sparse=True,
                )

            if is_trainer_rank:
                _append_worker_debug(cfg, rank, f"before_dense_fwd step={step}")
                with timer.gpu("dense_fwd_ms"):
                    loss, _ = compute_dense_loss(
                        cfg, dense_module, criterion,
                        dense_features, embedded_sparse, labels)
                row["loss"] = float(loss.detach().float().cpu().item())

                _append_worker_debug(cfg, rank, f"before_backward step={step}")
                with timer.gpu("backward_ms"):
                    embedded_sparse_grad = run_hybrid_backward(
                        loss=loss,
                        embedded_sparse=embedded_sparse,
                        dense_module=dense_module,
                        torch=torch,
                        device=device,
                    )

                _append_worker_debug(cfg, rank, f"before_optimizer step={step}")
                with timer.gpu("dense_optimizer_ms"):
                    dense_optimizer.step()
                    dense_optimizer.zero_grad(set_to_none=True)
            else:
                row["dense_fwd_ms"] = 0.0
                row["backward_ms"] = 0.0
                row["dense_optimizer_ms"] = 0.0
                embedded_sparse_grad = torch.zeros_like(embedded_sparse)

            embedded_sparse_grad = embedded_sparse_grad.contiguous()

            _append_worker_debug(cfg, rank, f"before_sparse_update step={step}")
            with timer.gpu("sparse_optimizer_ms"):
                if fair_remote_mode and use_dist:
                    dist.broadcast(embedded_sparse_grad, src=0)
                embedded_sparse_source.backward(
                    embedded_sparse_grad.to(embedded_sparse_source.device)
                )
                sparse_optimizer.step()
                sparse_optimizer.zero_grad(set_to_none=True)

            if profiler is not None:
                profiler.step()

            # GPU stages are timed with CUDA events and resolved in finish() after
            # a single device drain, so no stage absorbs a neighbor's un-drained
            # tail. finish() returns that drain wait (the cross-rank straggler
            # cost) instead of letting it vanish into the step_total gap.
            row["step_sync_wait_ms"] = timer.finish()
            row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
            row["collective_launch_ms"] = 0.0
            row["collective_wait_ms"] = (
                row["embed_lookup_local_ms"] if use_dist else 0.0
            )
            rows.append(finalize_torchrec_row(row))
            _append_worker_debug(cfg, rank, f"before_step_barrier step={step}")
            _barrier_for_step_alignment(
                dist=dist,
                device=device,
                local_rank=local_rank,
                use_dist=use_dist,
            )
            _append_worker_debug(cfg, rank, f"after_step_barrier step={step}")

    _append_worker_debug(cfg, rank, f"before_write_rows count={len(rows)} out_csv={out_csv}")
    _write_rows(out_csv, rows)
    _append_worker_debug(cfg, rank, "after_write_rows")
    if (is_dist or use_uvm_caching) and dist.is_initialized():
        _append_worker_debug(cfg, rank, "before_barrier")
        if is_dist:
            dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
        _append_worker_debug(cfg, rank, "after_barrier")
        dist.destroy_process_group()
        _append_worker_debug(cfg, rank, "after_destroy_process_group")
    return rows


class TorchRecRunner(BenchmarkRunner):
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _rank_output_dir(self, cfg: RunConfig) -> Path:
        return Path(cfg.output_root) / "outputs" / cfg.run_id / "torchrec_ranks"

    def _build_torchrun_cmd(self, repo_root: Path, cfg: RunConfig, config_json: Path) -> list[str]:
        # The whole resolved config (incl. model_args) is handed to each worker
        # as JSON, so the launcher argv stays tiny and schema-drift free.
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
        rows = _run_single_or_dist_worker(
            repo_root=repo_root,
            cfg=cfg,
            rank=0,
            world_size=1,
            local_rank=0,
            out_csv=Path(cfg.torchrec_main_csv),
        )
        return {"backend": "torchrec", "rows": rows}

    def _run_distributed(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rank_dir = self._rank_output_dir(cfg)
        ensure_shared_dir(rank_dir)
        _remove_stale_distributed_outputs(cfg, rank_dir)

        worker_cfg = dataclasses.replace(cfg, start_server=False)
        config_json = dump_run_config(worker_cfg, rank_dir / "worker_config.json")
        cmd = self._build_torchrun_cmd(repo_root, cfg, config_json)

        env = os.environ.copy()
        env["RS_DEMO_TORCHREC_WORKER"] = "1"
        env["RS_DEMO_TORCHREC_WORKER_DIR"] = str(rank_dir)
        socket_ifname = _pick_socket_ifname()
        if socket_ifname:
            env.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
            env.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
        res = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                "torchrun worker failed\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}"
            )

        world_size = cfg.nnodes * cfg.nproc_per_node
        rank_csvs = [rank_dir / f"rank{rank}.csv" for rank in range(world_size)]
        missing = [str(path) for path in rank_csvs if not path.exists()]
        if missing:
            raise RuntimeError(f"missing rank csv outputs: {missing}")
        rows = _merge_rank_outputs(rank_csvs, Path(cfg.torchrec_main_csv))
        return {"backend": "torchrec", "rows": rows}

    def run(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        if cfg.backend != "torchrec":
            raise ValueError("TorchRecRunner requires cfg.backend to be 'torchrec'.")
        validate_torchrec_config(cfg)
        if cfg.steps <= 0:
            raise ValueError("TorchRec runner requires --steps to be greater than 0.")

        ensure_torchrec_available()

        if os.environ.get("RS_DEMO_TORCHREC_WORKER") == "1":
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(
                os.environ.get("WORLD_SIZE", str(cfg.nnodes * cfg.nproc_per_node))
            )
            worker_dir = Path(os.environ["RS_DEMO_TORCHREC_WORKER_DIR"])
            ensure_shared_dir(worker_dir)
            out_csv = worker_dir / f"rank{rank}.csv"
            rows = _run_single_or_dist_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                out_csv=out_csv,
            )
            return {"backend": "torchrec", "rows": rows}

        if cfg.nnodes * cfg.nproc_per_node <= 1 and cfg.torchrec_memory_mode == "hbm":
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
