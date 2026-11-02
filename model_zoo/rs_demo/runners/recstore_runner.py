from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
_repo_root = Path(__file__).resolve().parents[3]
_pytorch_src = str(_repo_root / "src" / "python" / "pytorch")
if _pytorch_src not in sys.path:
    sys.path.insert(0, _pytorch_src)

from ..config import (
    RunConfig,
    ensure_shared_dir,
    resolve_num_embeddings_per_feature,
    validate_recstore_config,
)
from ..data.dlrm_source import (
    build_kjt_batch_from_dense_sparse_labels,
    convert_kjt_ids_to_fused_ids,
    convert_kjt_ids_to_fused_ids_device,
    get_default_cat_names,
    inject_project_paths,
)
from ..models.dlrm import (
    build_criterion,
    build_dense_module,
    compute_dense_loss,
)
from ..models.utils import (
    prepare_hybrid_dlrm_input,
    reshape_torchrec_embeddings_for_dlrm,
)
from python.pytorch.recstore.benchmark.report import finalize_recstore_row
from python.pytorch.recstore.analysis.profiler import (
    ProfilerConfig,
    build_torchrec_profiler,
)
from ..runtime.timing import StepTimer
from ..runtime.continuous_loop import ContinuousLoop, continuous_enabled, verify_continuous_protocol
from ..runtime.worker_common import (
    barrier_for_step_alignment as _barrier_for_step_alignment,
    bool_int as _bool_int,
    read_worker_context as _read_worker_context,
    write_rows as _write_rows,
)
from .base import (
    BenchmarkRunner,
    build_rank_dataloader as _build_train_dataloader_for_mode,
    wrap_dense_module_for_dist as _maybe_wrap_dense_module_for_dist,
)

from recstore.embedding_read_path import (
    BagPipeReadPath,
    PreparedTicket,
    build_embedding_read_path,
)
from recstore.optim import OptimizationPluginRegistry


_RECSTORE_DEFER_OPS_LOAD = "RECSTORE_DEFER_OPS_LOAD"

os.environ.setdefault(_RECSTORE_DEFER_OPS_LOAD, "1")
import recstore


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


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


def _finalize_step_timing(row: dict[str, Any], *, wall_start: float) -> None:
    total_ms = (time.perf_counter() - wall_start) * 1e3
    row["step_total_ms"] = total_ms
    row["step_end_to_end_ms"] = total_ms
    row["samples_per_sec"] = _safe_ratio(row["batch_size"] * 1000.0, total_ms)
    row["batches_per_sec"] = _safe_ratio(1000.0, total_ms)


def _consume_perf_stats(obj: Any) -> dict[str, float]:
    stats = obj.consume_perf_stats(reset=True)
    return stats if isinstance(stats, dict) else {}


def _merge_consumed_perf_stats(row: dict[str, Any], stats: dict[str, float]) -> None:
    for key, value in stats.items():
        if row.get(key):
            continue
        row[key] = value


def _reset_perf_stats(obj: Any) -> None:
    obj.reset_perf_stats()


class _EventLog:
    """Append-only JSONL host timeline used to align traces with wall clock."""

    def __init__(self, path: str, rank: int) -> None:
        self._file = None
        if not path:
            return
        target = Path(path)
        target = target.with_name(f"{target.stem}.rank{rank}{target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._file = target.open("w", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        if self._file is None:
            return
        record = {
            "event": event,
            "epoch": time.time(),
            "mono_ns": time.monotonic_ns(),
            **fields,
        }
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _iter_profiled_steps(steps: Any, profiler_context: Any) -> Any:
    """Yield steps while the profiler context is active.

    A generator keeps the profiling window tied to the training loop without
    re-indenting the whole loop body.
    """
    with profiler_context:
        yield from steps


def _make_recstore_trace_handler(cfg: RunConfig, rank: int):
    def _handler(prof: Any) -> None:
        trace_path = Path(cfg.recstore_trace_dir) / f"rank{rank}.pt.trace.json"
        prof.export_chrome_trace(str(trace_path))

    return _handler


def _fill_prefetch_buffer(
    prepared_batches: deque,
    prepare_fn: Any,
    *,
    from_step: int,
    target_buffer: int,
    max_steps: int,
) -> None:
    """Prepare batches until the buffer exceeds target_buffer or steps run out."""
    current = len(prepared_batches)
    needed = target_buffer + 1 - current
    for i in range(needed):
        future_step = from_step + current + i
        if future_step >= max_steps:
            break
        prepared_batches.append(prepare_fn(future_step))


def _maybe_warmup_gpu_local_shm_fast_path(
    cfg: RunConfig, client: Any, device: torch.device
) -> bool:
    if cfg.nnodes != 1 or cfg.single_node_ps_backend != "local_shm":
        return False
    if device.type != "cuda":
        return False
    if not client.is_shared_local_shm_table():
        return False
    # activate_shard exists on test/distributed fakes, not RecStoreClient.
    if hasattr(client, "activate_shard"):
        client.activate_shard(0)
    if client.current_ps_backend() != "local_shm":
        client.set_ps_backend("local_shm")
    return bool(client.warmup_local_lookup_flat_cuda_region())


class RecStoreRunner(BenchmarkRunner):
    """Train a DLRM-style model with embeddings served by the RecStore PS.

    A single run is one of: an in-process single worker, a torchrun launcher
    (re-invokes this module per rank and merges their CSVs), or one distributed
    worker.  Per step it times lookup -> dense fwd/bwd -> dense opt -> sparse
    update. Embedding reads follow ``cfg.read_mode`` (direct / prefetch).
    """

    backend = "recstore"
    rank_dir_name = "recstore_ranks"

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _main_csv(self, cfg: RunConfig) -> Path:
        return Path(cfg.recstore_main_csv)

    def _after_torchrun(self, cfg: RunConfig, rank_dir: Path, result) -> None:
        # [BagPipe-probe] keep worker stderr even on success: C++ WARNINGs
        # (GPU cache fallbacks, update failures) are otherwise lost.
        try:
            (rank_dir / "worker_stderr.log").write_text(
                result.stderr or "", encoding="utf-8"
            )
        except OSError:
            pass

    def _run_single_process(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        return self._run_local_worker(
            repo_root=repo_root, cfg=cfg, rank=0, world_size=1, local_rank=0,
            out_csv=Path(cfg.recstore_main_csv),
        )

    # -- worker setup ------------------------------------------------------

    def _init_process_group(self, world_size, local_rank, dist):
        """Set up the CUDA device and (if distributed) the process group."""
        use_dist = world_size > 1
        backend = "nccl"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if use_dist and not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                device_id=device if device.type == "cuda" else None,
            )
            dist.barrier()
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

    def _build_read_path(
        self,
        cfg: RunConfig,
        client: Any,
        embedding_module: Any,
        eb_configs: list[dict],
        table_offsets: dict[str, int],
        default_cat_names: list[str],
        device: torch.device,
    ) -> tuple[Any, Any, Any, bool]:
        """Build the embedding read path, its optimizer, and the BagPipe plugin.

        Returns ``(plugin, read_path, sparse_optimizer, use_planned_bagpipe)``.
        """
        use_bagpipe = cfg.optimization.plugin == "bagpipe" or cfg.read_mode == "bagpipe"
        use_planned_bagpipe = use_bagpipe and cfg.optimization.prescan

        if cfg.read_mode == "ps_pipeline":
            from recstore.ps_pipeline import PSPipeline, PSPipelineSGD
            canonical_types = None
            if cfg.ps_pipeline_canonical_plan:
                import importlib.metadata
                import json
                required = {"torch": "2.7.1", "torchrec": "1.2.0", "fbgemm-gpu": "1.2.0"}
                if any(importlib.metadata.version(name).split("+")[0] != version
                       for name, version in required.items()):
                    raise RuntimeError("Canonical PS requires the validated Torch 2.7.1/FBGEMM 1.2.0 stack")
                with open(cfg.ps_pipeline_canonical_plan) as plan_file:
                    reference_plan = json.load(plan_file)
                expected_tables = {"t_" + feature for feature in table_offsets}
                if set(reference_plan) != expected_tables:
                    raise ValueError("Canonical reference plan must cover exactly the model tables")
                allowed = {("table_wise", "fused"): 0, ("data_parallel", "dense"): 1}
                canonical_types = []
                for feature in table_offsets:
                    entry = reference_plan["t_" + feature]
                    pair = (entry.get("sharding_type"), entry.get("compute_kernel"))
                    if pair not in allowed:
                        raise ValueError("Unsupported canonical reference table sharding/kernel")
                    canonical_types.append(allowed[pair])
            pipeline_class, owner_options = PSPipeline, {}
            owner_forward = os.environ.get("RECSTORE_PS_OWNER_FORWARD") == "1"
            owner_gpu_plan = os.environ.get("RECSTORE_PS_CANONICAL_GPU_PLAN") == "1"
            owner_handoff = os.environ.get("RECSTORE_PS_OWNER_CANDIDATE_HANDOFF") == "1"
            owner_verify = os.environ.get("RECSTORE_PS_OWNER_VERIFY_HANDOFF") == "1"
            owner_compute_ahead = os.environ.get("RECSTORE_PS_OWNER_COMPUTE_AHEAD") == "1"
            metadata_depth = int(os.environ.get("RECSTORE_PS_METADATA_LOOKAHEAD", "0"))
            if metadata_depth not in (0, 4, 8) or (metadata_depth and not owner_forward):
                raise ValueError("Metadata lookahead requires unified owner and depth 0/4/8")
            if metadata_depth and os.environ.get("RECSTORE_PS_PREPARE_DURING_DENSE") == "1":
                raise ValueError("Metadata lookahead keeps the original one-next adoption point")
            if owner_compute_ahead and not (owner_forward and owner_handoff):
                raise ValueError("Owner compute-ahead requires unified candidate handoff")
            if owner_verify and not owner_handoff:
                raise ValueError("Diagnostic handoff verification requires candidate handoff")
            if owner_handoff and not owner_forward:
                raise ValueError("Acknowledged candidate handoff requires unified owner forward")
            if owner_gpu_plan and os.environ.get("RECSTORE_PS_OWNER_GPU") != "1":
                raise ValueError("Canonical GPU plan staging requires GPU owner updates")
            if owner_forward and os.environ.get("RECSTORE_PS_OWNER_GPU") != "1":
                raise ValueError("Unified owner forward requires GPU owner updates")
            if os.environ.get("RECSTORE_PS_OWNER_GPU") == "1":
                if not cfg.dense_cuda_graph or cfg.ps_type != "RDMA":
                    raise ValueError("GPU owner updates currently require RDMA and the dense CUDA graph")
                from recstore.ps_owner_pipeline import GPUOwnerPipeline
                pipeline_class = GPUOwnerPipeline
                owner_options["unified_forward"] = owner_forward
                owner_options["canonical_gpu_plan"] = owner_gpu_plan
                owner_options["candidate_handoff"] = owner_handoff
                owner_options["verify_handoff"] = owner_verify
                owner_options["compute_ahead"] = owner_compute_ahead
                owner_options["metadata_lookahead"] = metadata_depth
                owner_options["owner_table_end"] = (
                    list(table_offsets.values())[-1] + int(eb_configs[-1]["num_embeddings"]))
            read_path = pipeline_class(
                embedding_module, client, table_offsets=table_offsets,
                master_table_name=eb_configs[0]["name"],
                embedding_dim=cfg.embedding_dim, device=device,
                validation_state=cfg.validation_state,
                dependency_split=cfg.ps_pipeline_dependency_split, total_steps=cfg.steps,
                canonical_table_types=canonical_types,
                **owner_options,
            )
            return None, read_path, PSPipelineSGD(embedding_module, read_path), False
        if use_bagpipe:
            fused_id_prefixes = torch.tensor(
                [table_offsets[name] for name in default_cat_names],
                dtype=torch.int64,
                device=device,
            )

            def _id_extractor(sparse_features):
                values = sparse_features.values().to(torch.int64)
                try:
                    # DLRM has one ID per feature per sample, so a
                    # broadcast add avoids repeat_interleave and length
                    # reduction on the enqueue path.
                    return (
                        values.view(len(fused_id_prefixes), -1)
                        + fused_id_prefixes[:, None]
                    ).reshape(-1).contiguous()
                except RuntimeError:
                    return convert_kjt_ids_to_fused_ids_device(
                        sparse_features, table_offsets
                    )

            cache_capacity = (
                cfg.optimization.cache_capacity
                or cfg.gpu_cache_capacity
                or 160_000
            )
            print(
                "[rs_demo] BagPipe cache configured: "
                f"capacity={cache_capacity}, dim={cfg.embedding_dim}"
            )
            # BagPipe owns its cache policy.  The cold-start lookup must not
            # trip the ops layer's low-hit bypass latch and clear the cache.
            disable_bypass = getattr(
                client, "set_gpu_cache_lookup_bypass_enabled", None
            )
            if callable(disable_bypass):
                disable_bypass(False)
            # BagPipe also owns residency across CPU-side update paths.
            disable_clear_after_update = getattr(
                client, "set_clear_gpu_cache_after_cpu_update", None
            )
            if callable(disable_clear_after_update):
                disable_clear_after_update(False)

            master_table_name = eb_configs[0]["name"] if eb_configs else ""
            plugin = OptimizationPluginRegistry.create(
                "bagpipe",
                prescan=cfg.optimization.prescan,
                aggregation=cfg.optimization.aggregation,
                embedding_module=embedding_module,
                kv_client=client,
                lookahead=cfg.optimization.lookahead,
                cleanup_proportion=cfg.optimization.cleanup_proportion,
                cache_capacity=cache_capacity,
                embedding_dim=cfg.optimization.embedding_dim,
                fuse_k=cfg.fuse_k,
                table_offsets=table_offsets,
                table_sizes={
                    cfg_item["feature_names"][0]: cfg_item["num_embeddings"]
                    for cfg_item in eb_configs
                },
                master_table_name=master_table_name,
                device=device,
                lr=0.01,
                id_extractor=_id_extractor,
            )
            sparse_optimizer = plugin.create_sparse_optimizer(
                [embedding_module], lr=0.01
            )
            read_path = BagPipeReadPath(plugin, device=device)
        else:
            fused_id_offsets = torch.tensor(
                [table_offsets[name] for name in default_cat_names],
                dtype=torch.int64,
            )
            plugin = None
            read_path = build_embedding_read_path(
                cfg.read_mode,
                embedding_module=embedding_module,
                prefetch_depth=cfg.prefetch_depth,
                embedding_dim=cfg.embedding_dim,
                feature_offsets=fused_id_offsets,
            )
            sparse_optimizer = recstore.SparseSGD([embedding_module], lr=0.01)

        return plugin, read_path, sparse_optimizer, use_planned_bagpipe

    def _build_training_modules(
        self,
        cfg: RunConfig,
        device: torch.device,
        local_rank: int,
        use_dist: bool,
        num_sparse_features: int,
    ) -> tuple[Any, Any, Any, Any]:
        """Build the dense model, criterion, and dense optimizer."""
        dense_module = build_dense_module(
            cfg,
            num_sparse_features=num_sparse_features,
            embedding_dim=cfg.embedding_dim,
            device=device,
        )
        if cfg.dense_cuda_graph:
            # DDP hooks remain after unwrapping; never install them for capture.
            if use_dist:
                with torch.no_grad():
                    for value in list(dense_module.parameters()) + list(dense_module.buffers()):
                        torch.distributed.broadcast(value, src=0)
        else:
            dense_module = _maybe_wrap_dense_module_for_dist(
                dense_module=dense_module, device=device,
                local_rank=local_rank, use_dist=use_dist,
            )
        unwrapped_module = (
            dense_module.module
            if isinstance(dense_module, torch.nn.parallel.DistributedDataParallel)
            else dense_module
        )
        criterion = build_criterion(cfg, unwrapped_module)
        dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)
        return dense_module, unwrapped_module, criterion, dense_optimizer

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

        overlap_dense = os.environ.get("RECSTORE_DENSE_REDUCE_OVERLAP") == "1"
        if overlap_dense and (not cfg.dense_cuda_graph or cfg.read_mode != "ps_pipeline"
                              or cfg.torchrec_dist_mode == "fair_remote"):
            raise ValueError("Dense overlap requires PS pipeline CUDA graph and no fair_remote")
        prepare_during_dense = os.environ.get("RECSTORE_PS_PREPARE_DURING_DENSE") == "1"
        if prepare_during_dense and not (
            cfg.read_mode == "ps_pipeline" and cfg.ps_pipeline_dependency_split
            and cfg.dense_cuda_graph and overlap_dense
        ):
            raise ValueError("Prepare during dense requires dependency-split PS CUDA graph and dense overlap")
        default_cat_names = get_default_cat_names()
        print(f"[rs_demo] repo_root={repo_root}")
        print("[rs_demo] backend=recstore")
        if cfg.read_mode == "direct" and cfg.prefetch_depth > 0:
            print(
                "[rs_demo] read_mode=direct ignores prefetch_depth="
                f"{cfg.prefetch_depth}"
            )

        orig_cwd = Path.cwd()
        plugin = None
        events: _EventLog | None = None
        step_iter = None
        continuous = continuous_error = None
        try:
            os.chdir(str(self.runtime_dir))
            torch.manual_seed(cfg.seed)
            device, use_dist = self._init_process_group(
                world_size, local_rank, dist
            )

            verify_continuous_protocol(torch, dist, device, world_size, cfg.steps)
            continuous_mode = continuous_enabled(cfg, device, world_size, lane="recstore")
            recstore.load_ops_library()
            client = recstore.RecStoreClient()
            if cfg.ps_type.upper() in {"LOCAL_SHM", "HIERKV"}:
                client.set_ps_backend(cfg.single_node_ps_backend)
            elif cfg.ps_type.upper() == "RDMA":
                client.set_ps_backend("rdma")

            dataset, dataloader = _build_train_dataloader_for_mode(repo_root, cfg, rank)

            eb_configs, table_offsets, RecStoreEmbeddingBagCollection = (
                self._build_embedding_module(cfg, client, default_cat_names)
            )
            embedding_init_start = time.perf_counter()
            embedding_module = RecStoreEmbeddingBagCollection(
                embedding_bag_configs=eb_configs,
                enable_fusion=cfg.recstore_enable_fusion,
                fusion_k=cfg.fuse_k,
                kv_client=client,
                initialize_tables=(rank == 0),
                shared_fused_namespace=(cfg.read_mode == "ps_pipeline"),
            )
            embedding_init_ms = (time.perf_counter() - embedding_init_start) * 1e3
            # -- optimization plugin + read path --------------------------------
            use_bagpipe = cfg.optimization.plugin == "bagpipe" or cfg.read_mode == "bagpipe"
            use_ps_pipeline = cfg.read_mode == "ps_pipeline"
            plugin, read_path, sparse_optimizer, use_planned_bagpipe = (
                self._build_read_path(
                    cfg=cfg,
                    client=client,
                    embedding_module=embedding_module,
                    eb_configs=eb_configs,
                    table_offsets=table_offsets,
                    default_cat_names=default_cat_names,
                    device=device,
                )
            )

            if continuous_mode and not all(getattr(read_path, name, False) for name in (
                "unified_forward", "canonical_gpu_plan", "candidate_handoff", "compute_ahead"
            )):
                raise ValueError("Continuous PS only supports the selected owner forward/handoff/compute-ahead path")
            if continuous_mode and getattr(read_path, "metadata_lookahead", 0) != 4:
                raise ValueError("Continuous PS initially supports the selected H4 metadata queue only")
            _barrier_for_step_alignment(
                dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
            )

            dense_module, unwrapped_module, criterion, dense_optimizer = (
                self._build_training_modules(
                    cfg=cfg,
                    device=device,
                    local_rank=local_rank,
                    use_dist=use_dist,
                    num_sparse_features=len(default_cat_names),
                )
            )
            record_pooled_grad = getattr(embedding_module, "record_pooled_grad", None)

            if _maybe_warmup_gpu_local_shm_fast_path(cfg=cfg, client=client, device=device):
                print("[rs_demo] warmed local_shm lookup payload region for GPU fast path")
                _barrier_for_step_alignment(
                    dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
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
            lifecycle = {"rank": rank, "embedding_init_ms": embedding_init_ms,
                "dense_reduce_overlap_enabled": int(overlap_dense),
                "ps_pipeline_prepare_during_dense_enabled": int(prepare_during_dense),
                "dense_reduce_overlap_groups": {"dense": "WORLD" if use_dist else "local",
                    "sparse": ("GPU owner NCCL/WORLD + PS exact writeback" if
                               os.environ.get("RECSTORE_PS_OWNER_GPU") == "1" else "PS/RDMA, no NCCL"),
                    "dense_buffer_group": "None/default"},
                "gradient_communication_semantics": "remaining_join" if overlap_dense else "full_reduce"}
            lifecycle.update(timing_metadata)
            if use_ps_pipeline:
                lifecycle["embedding_namespace"] = embedding_module.shared_fused_namespace_identity()
                lifecycle["owned_read_native"] = callable(getattr(client.ops, "emb_read_into", None))
                lifecycle["ps_pipeline_dependency_split"] = cfg.ps_pipeline_dependency_split
                lifecycle["ps_pipeline_canonical_plan"] = cfg.ps_pipeline_canonical_plan
                lifecycle["ps_pipeline_canonical_types"] = list(read_path._canonical_table_types)
                if cfg.ps_pipeline_dependency_split:
                    lifecycle["ps_pipeline_metadata_group_init_ms"] = read_path.metadata_group_init_ms
                    lifecycle["ps_pipeline_metadata_byte_semantics"] = "padded tensor payload per peer; excludes transport framing"
            dense_graph = dense_gradient_buffer = None
            if cfg.dense_cuda_graph:
                from ..runtime.dense_graph import CapturedDlrmDense, DenseGradientBuffer
                local_samples = len(dataloader.batch_sampler.sampler)
                batch_sizes = {min(cfg.batch_size, local_samples)}
                tail = local_samples % cfg.batch_size
                if tail and not dataloader.drop_last:
                    batch_sizes.add(tail)
                dense_gradient_buffer = DenseGradientBuffer(dense_module)
                dense_graph = CapturedDlrmDense(
                    dense_module, criterion, dense_gradient_buffer,
                    batch_sizes=sorted(batch_sizes), num_features=len(default_cat_names),
                    embedding_dim=cfg.embedding_dim,
                )
                lifecycle.update(
                    dense_graph_capture_ms=dense_graph.capture_ms,
                    dense_graph_batch_sizes=list(dense_graph.batch_sizes),
                    dense_graph_memory=dense_graph.memory_stats,
                    dense_graph_scope="dense forward/backward; individual stage times unavailable",
                )
                _barrier_for_step_alignment(
                    dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
                )
            # Keep the complete access schedule on CPU, including epoch transitions.
            if (
                use_planned_bagpipe
                and plugin is not None
                and plugin.is_enabled
            ):
                prescan_start = time.perf_counter()
                prescan_batches = 0
                prescan_iter = iter(dataloader)
                for _ in range(cfg.steps):
                    try:
                        dense_b, sparse_b, _labels_b = next(prescan_iter)
                    except StopIteration:
                        prescan_iter = iter(dataloader)
                        dense_b, sparse_b, _labels_b = next(prescan_iter)
                    _, sf = build_kjt_batch_from_dense_sparse_labels(
                        dense_b, sparse_b, _labels_b, device=device
                    )
                    plugin.prescan_batch(prescan_batches, sf)
                    prescan_batches += 1
                plugin.finalize_prescan()
                # Replay the same seeded sampling sequence, including
                # RandomSampler epoch changes in single-rank training.
                del prescan_iter
                dataset, dataloader = _build_train_dataloader_for_mode(repo_root, cfg, rank)
                _barrier_for_step_alignment(
                    dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
                )
                lifecycle["bagpipe_prescan_total_ms"] = (time.perf_counter() - prescan_start) * 1e3
                lifecycle.update(plugin._controller.consume_stats(reset=False))
                print(
                    f"[rs_demo] BagPipe oracle prescan: {prescan_batches} batches "
                    f"in {time.perf_counter() - prescan_start:.1f}s"
                )

            rows: list[dict[str, Any]] = []
            data_iter_state = {"iter": iter(dataloader)}
            prepared_batches: deque = deque()
            metadata_depth = getattr(read_path, "metadata_lookahead", 0)
            cpu_metadata_handles = {}  # Bounded CPU inputs only, never parameter values.
            metadata_loop_start = None
            if metadata_depth:
                lifecycle["metadata_lookahead_depth"] = metadata_depth
                lifecycle["metadata_lookahead_capacity"] = metadata_depth + 1
                lifecycle["lookup_ids_build_scope"] = "background CPU ticket service; not main wait"
                lifecycle["metadata_cpu_byte_scope"] = "buffered logical payload; aliases may count twice; excludes worker scratch"
            skip_kjt = os.environ.get("RECSTORE_PS_SKIP_KJT") == "1"
            if skip_kjt and not use_ps_pipeline:
                raise ValueError("Skipping KJT requires PS pipeline")
            if os.environ.get("RECSTORE_PS_PREPARE_ASYNC") == "1":
                raise ValueError("Async CPU preparation is not enabled in this composition")

            def prepare_next_batch(batch_step: int):
                events.emit("prepare_start", step=batch_step)
                row: dict[str, Any] = {
                    "rank": rank,
                    "batch_size": cfg.batch_size,
                    "step": batch_step,
                    "warmup_excluded": _bool_int(batch_step < cfg.warmup_steps),
                    "ps_owner_forward_enabled": int(getattr(read_path, "unified_forward", False)),
                    "ps_owner_canonical_gpu_plan_enabled": int(getattr(read_path, "canonical_gpu_plan", False)),
                    "ps_owner_candidate_handoff_enabled": int(getattr(read_path, "candidate_handoff", False)),
                    "ps_owner_verify_handoff_enabled": int(getattr(read_path, "verify_handoff", False)),
                    "ps_owner_compute_ahead_enabled": int(getattr(read_path, "compute_ahead", False)),
                    "ps_metadata_lookahead_depth": metadata_depth,
                }
                batch_prepare_start = time.perf_counter()
                try:
                    dense_batch, sparse_batch, labels_batch = next(data_iter_state["iter"])
                except StopIteration:
                    data_iter_state["iter"] = iter(dataloader)
                    dense_batch, sparse_batch, labels_batch = next(data_iter_state["iter"])
                row["batch_prepare_ms"] = (time.perf_counter() - batch_prepare_start) * 1e3
                row["batch_size"] = int(dense_batch.shape[0])

                input_pack_start = time.perf_counter()
                if skip_kjt:
                    sparse_features = None
                else:
                    _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                        dense_batch, sparse_batch, labels_batch,
                        device=None if use_ps_pipeline else device,
                    )
                row["input_pack_ms"] = (time.perf_counter() - input_pack_start) * 1e3

                if metadata_depth:
                    ticket = read_path.queue_cpu_metadata(batch_step, sparse_batch)
                    cpu_metadata_handles[batch_step] = ticket
                else:
                    ticket = read_path.on_batch_prepared(
                        batch_step, sparse_features, sparse_batch, row)
                if metadata_depth:
                    pass  # ID statistics are attached after main-thread adoption.
                elif isinstance(ticket, PreparedTicket):
                    _add_sparse_id_stats(
                        row,
                        sparse_features,
                        table_offsets,
                        precomputed=(
                            int(ticket.unique_ids.numel()),
                            ticket.raw_count,
                        ),
                    )
                elif (
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
                events.emit(
                    "prepare_end",
                    step=batch_step,
                    scope="cpu_inputs_and_metadata_enqueue" if metadata_depth else "batch_prepare",
                    metadata_pending=bool(metadata_depth and not ticket.future.done()),
                    enqueue_ms=row.get("bagpipe_enqueue_ms"),
                    prepare_ms=row.get("batch_prepare_ms", 0.0)
                    + row.get("input_pack_ms", 0.0),
                )
                return (
                    batch_step, row,
                    dense_batch, sparse_features, labels_batch, ticket,
                )

            def adopt_metadata_batch(batch):
                batch_step, row, dense, features, labels, ticket = batch
                if batch_step >= read_path._prepared:
                    ticket = read_path.adopt_cpu_metadata(batch_step, cpu_metadata_handles[batch_step], row)
                    _add_sparse_id_stats(row, features, table_offsets,
                                         precomputed=(int(ticket[0].numel()), int(ticket[2])))
                return batch_step, row, dense, features, labels, ticket

            def metadata_buffer_stats(row):
                inputs = plans = completed = 0
                for entry in prepared_batches:
                    handle = cpu_metadata_handles[entry[0]]
                    inputs += sum(x.numel()*x.element_size() for x in (entry[2], entry[4], handle.source))
                    if handle.future.done():
                        plans += handle.future.result()[7]
                        completed += 1
                row.update(ps_metadata_cpu_queued=len(cpu_metadata_handles),
                           ps_metadata_cpu_completed=completed,
                           ps_metadata_cpu_queue_hwm=read_path._metadata_queue_hwm,
                           ps_metadata_cpu_input_bytes=inputs,
                           ps_metadata_cpu_plan_payload_bytes=plans)

            profiler = build_torchrec_profiler(
                ProfilerConfig(
                    enabled=cfg.recstore_profiler,
                    trace_dir=cfg.recstore_trace_dir,
                    warmup=cfg.recstore_profiler_warmup,
                    active=cfg.recstore_profiler_active,
                    repeat=cfg.recstore_profiler_repeat,
                ),
                on_trace_ready=(
                    _make_recstore_trace_handler(cfg, rank)
                    if cfg.recstore_profiler
                    else None
                ),
            )
            events = _EventLog(cfg.recstore_event_log, rank)
            events.emit(
                "profiler_begin",
                rank=rank,
                profiler_enabled=bool(profiler),
                warmup=cfg.recstore_profiler_warmup,
                active=cfg.recstore_profiler_active,
                repeat=cfg.recstore_profiler_repeat,
            )
            step_iter = _iter_profiled_steps(
                range(cfg.steps), profiler or nullcontext()
            )
            if continuous_mode:
                continuous = ContinuousLoop(torch, device, cfg.steps)
                continuous.start()
            for step in step_iter:
                step_wall_start = time.perf_counter()
                backpressure_ms = continuous.admit(step) if continuous is not None else 0.0
                if metadata_depth and metadata_loop_start is None:
                    metadata_loop_start = step_wall_start  # Includes step-0 initial fill.
                events.emit("step_start", step=step)
                observed_depth = read_path.desired_buffer_size
                target_buffer = metadata_depth if metadata_depth else observed_depth
                _fill_prefetch_buffer(
                    prepared_batches, prepare_next_batch,
                    from_step=step, target_buffer=target_buffer, max_steps=cfg.steps,
                )
                if step + len(prepared_batches) >= cfg.steps:
                    read_path.advance_all()

                batch = prepared_batches.popleft()
                if metadata_depth:
                    batch = adopt_metadata_batch(batch)
                    cpu_metadata_handles.pop(step)  # Producer is complete before its source is released.
                (
                    _, row, dense_batch, sparse_features, labels_batch,
                    ticket,
                ) = batch

                row["ps_pipeline_prepare_during_dense_enabled"] = int(prepare_during_dense)
                if cfg.ps_pipeline_dependency_split:
                    row["ps_pipeline_lookahead_prepare_ms"] = 0.0
                if (cfg.ps_pipeline_dependency_split and step + 1 < cfg.steps
                        and not prepare_during_dense):
                    prepare_start = time.perf_counter()
                    if metadata_depth:
                        prepared_batches[0] = adopt_metadata_batch(prepared_batches[0])
                    else:
                        prepared_batches.append(prepare_next_batch(step + 1))
                    row["ps_pipeline_lookahead_prepare_ms"] = (time.perf_counter() - prepare_start) * 1e3

                if metadata_depth:
                    metadata_buffer_stats(row)
                _reset_perf_stats(embedding_module)
                sparse_optimizer.zero_grad()
                row.update(timing_metadata)
                timer = StepTimer(row, torch, device, mode="none" if continuous_mode else cfg.torchrec_timing_sync_mode)
                row["dense_reduce_overlap_enabled"] = int(overlap_dense)
                row["gradient_communication_enqueue_ms"] = 0.0
                row["gradient_communication_semantics"] = "remaining_join" if overlap_dense else "full_reduce"
                # embed_lookup and sparse_update hit the PS over the network (host
                # + network work) so they stay on the wall clock; the pure-GPU
                # dense stages use CUDA events via timer.gpu().
                events.emit("consume_start", step=step)
                with timer.cpu("embed_lookup_ms"):
                    read_path.before_lookup(step, sparse_features, ticket, row)
                    if use_ps_pipeline:
                        embeddings = read_path.lookup(sparse_features)
                    elif use_bagpipe:
                        embeddings = plugin._controller.lookup(sparse_features)
                    elif callable(record_pooled_grad):
                        with torch.no_grad():
                            embeddings = embedding_module(sparse_features)
                    else:
                        embeddings = embedding_module(sparse_features)
                events.emit("consume_end", step=step)

                if embeddings is None:
                    raise RuntimeError("recstore embedding module returned no embeddings")

                events.emit("pack_start", step=step)
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
                        synchronize=not cfg.dense_cuda_graph,
                    )
                if use_ps_pipeline and cfg.ps_pipeline_early_submit:
                    embedded_sparse.register_hook(sparse_optimizer.submit_pooled_grad)
                events.emit("pack_end", step=step)
                if dense_graph is not None:
                    events.emit("dense_graph_start", step=step)
                    with timer.gpu("dense_graph_ms"):
                        loss, embedded_sparse_grad = dense_graph(
                            dense_features, embedded_sparse, labels
                        )
                    events.emit("dense_graph_end", step=step)
                    row["dense_fwd_ms"] = row["backward_ms"] = ""
                    dense_work = None
                    if overlap_dense:
                        with timer.cpu("gradient_communication_enqueue_ms"):
                            dense_work = dense_gradient_buffer.submit_reduce()
                    # The flat dense buffer and owned sparse staging are disjoint.
                    try:
                        if prepare_during_dense and step + 1 < cfg.steps:
                            # Current read was consumed; the pipeline uses its completed
                            # _consumed_read. Prepare before submit advances _submitted.
                            with timer.cpu("ps_pipeline_lookahead_prepare_ms"):
                                prepared_batches.append(prepare_next_batch(step + 1))
                        with timer.cpu("sparse_graph_submit_ms"):
                            with timer.cpu("sparse_backward_replay_ms"):
                                sparse_optimizer.record_pooled_grad(embedded_sparse_grad)
                            with timer.cpu("sparse_optimizer_step_ms"):
                                sparse_optimizer.step()
                    except BaseException as sparse_error:
                        if dense_work is not None:
                            try:
                                dense_work.wait()
                            except BaseException as join_error:
                                sparse_error.add_note("Dense reduction join also failed: " + repr(join_error))
                        raise
                    with timer.gpu("gradient_communication_ms"):
                        # In overlap mode this event interval is only the remaining join.
                        if overlap_dense:
                            if dense_work is not None:
                                dense_work.wait()
                        else:
                            dense_gradient_buffer.reduce()
                else:
                    events.emit("fwd_start", step=step)
                    with timer.gpu("dense_fwd_ms"):
                        loss, _ = compute_dense_loss(
                            cfg, dense_module, criterion, dense_features, embedded_sparse, labels
                        )

                    events.emit("fwd_end", step=step)
                    events.emit("bwd_start", step=step)
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

                    events.emit("bwd_end", step=step)
                events.emit("dense_opt_start", step=step)
                with timer.gpu("dense_optimizer_ms"):
                    dense_optimizer.step()
                    dense_optimizer.zero_grad(set_to_none=True)

                events.emit("dense_opt_end", step=step)
                events.emit("sparse_start", step=step)
                with timer.cpu("sparse_optimizer_ms"):
                    if dense_graph is None:
                        replay_start = time.perf_counter()
                        sparse_grad = embedded_sparse_grad.to(embedded_sparse_source.device)
                        if use_ps_pipeline:
                            if not cfg.ps_pipeline_early_submit:
                                sparse_optimizer.record_pooled_grad(sparse_grad)
                        elif callable(record_pooled_grad):
                            prepared_ids = (
                                ticket
                                if isinstance(ticket, tuple) and len(ticket) == 3
                                else None
                            )
                            record_pooled_grad(
                                sparse_features, sparse_grad, prepared_ids=prepared_ids
                            )
                        else:
                            embedded_sparse_source.backward(sparse_grad)
                        row["sparse_backward_replay_ms"] = (time.perf_counter() - replay_start) * 1e3

                        optimizer_step_start = time.perf_counter()
                        sparse_optimizer.step()
                        row["sparse_optimizer_step_ms"] = (
                            time.perf_counter() - optimizer_step_start
                        ) * 1e3

                    # Overlap PS update latency by preparing future batches.
                    overlap_prepare_start = time.perf_counter()
                    while (
                        not metadata_depth and len(prepared_batches) <= observed_depth
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
                    row["sparse_optimizer_flush_ms"] = (
                        time.perf_counter() - flush_start
                    ) * 1e3
                    read_path.after_sparse_update(
                        step, sparse_features, sparse_optimizer, row
                    )
                    if use_bagpipe and step == cfg.steps - 1:
                        # Charge the final deferred SGD to training as well.
                        plugin._controller._wait_pending()
                    if use_ps_pipeline and step == cfg.steps - 1:
                        read_path._wait_pending()
                        if metadata_depth:
                            if cpu_metadata_handles or prepared_batches or read_path._prepared != cfg.steps:
                                raise RuntimeError("Metadata lookahead did not consume the exact tail")
                            # All producer results were adopted; include executor drain in final timing.
                            read_path._metadata_worker.shutdown(wait=True, cancel_futures=False)
                            lifecycle["metadata_final_drain_complete"] = True
                    sparse_optimizer.zero_grad()

                row["sparse_optimizer_ms"] += row.get("sparse_graph_submit_ms", 0.0)
                events.emit("sparse_end", step=step)
                if continuous is not None:
                    continuous.record_loss_and_terminal(step, loss)
                    row["loss"] = ""
                else:
                    row["loss"] = float(loss.detach().float().cpu().item())
                _merge_consumed_perf_stats(row, _consume_perf_stats(embedding_module))
                row["step_sync_wait_ms"] = timer.finish()
                # These stats are CPU-only. In continuous mode they describe
                # enqueue/protocol state, not completed GPU work.
                read_path.after_step_sync(row)
                row["dense_compute_ms"] = row["dense_optimizer_ms"] + (
                    row["dense_graph_ms"] if dense_graph is not None
                    else row["dense_fwd_ms"] + row["backward_ms"]
                )
                if continuous is None:
                    _finalize_step_timing(row, wall_start=step_wall_start)
                else:
                    continuous.finish_row(row, step_wall_start, backpressure_ms)
                rows.append(finalize_recstore_row(row))
                events.emit("barrier_start", step=step)
                if continuous is None:
                    _barrier_for_step_alignment(
                        dist=dist, device=device, local_rank=local_rank, use_dist=use_dist
                    )
                events.emit("barrier_end", step=step)

                if (step + 1) % 10 == 0:
                    timing_label = "host_enqueue" if continuous is not None else "step"
                    timing_value = rows[-1]["host_step_enqueue_wall_ms" if continuous is not None else "step_total_ms"]
                    print(
                        f"[rs_demo] {'computed step' if getattr(read_path, 'compute_ahead', False) else 'step'} {step + 1}/{cfg.steps} "
                        f"emb={rows[-1]['emb_stage_ms']:.2f}ms {timing_label}={timing_value:.2f}ms"
                    )
                events.emit(
                    "step_end", step=step, step_total_ms=row["step_total_ms"],
                    provisional=bool(getattr(read_path, "compute_ahead", False) and
                                     getattr(read_path, "_retired_step", -1) < step)
                )
                if profiler is not None:
                    profiler.step()
                if continuous is None:
                    rows[-1]["step_wall_ms"] = (time.perf_counter() - step_wall_start) * 1e3
                else:
                    rows[-1]["host_step_enqueue_wall_ms"] = (time.perf_counter() - step_wall_start) * 1e3

            if continuous is not None:
                loop_summary = continuous.finish(rows, lambda: _barrier_for_step_alignment(
                    dist=dist, device=device, local_rank=local_rank, use_dist=use_dist))
                loop_summary.update(rank=rank, backend="recstore", world_size=world_size,
                                    configured_warmup_steps=cfg.warmup_steps, run_id=cfg.run_id)
                lifecycle["continuous_loop"] = loop_summary
                loop_path = Path(cfg.output_root) / "outputs" / cfg.run_id / f"continuous_rank{rank}.json"
                loop_path.parent.mkdir(parents=True, exist_ok=True)
                loop_path.write_text(json.dumps(loop_summary, indent=2) + "\n")
            if metadata_depth:
                lifecycle["metadata_training_loop_ms"] = (time.perf_counter()-metadata_loop_start)*1e3
            print("[rs_demo] workload finished")

            if device.type == "cuda":
                lifecycle["training_cuda_memory"] = {
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
            _write_rows(out_csv, rows)
            if cfg.validation_state:
                if not (use_bagpipe or use_ps_pipeline):
                    raise RuntimeError("RecStore validation-state requires bagpipe or ps_pipeline")
                from ..runtime.validation import export_validation_state
                export_validation_state(
                    Path(cfg.output_root) / "outputs" / cfg.run_id / f"validation_rank{rank}.pt",
                    embedding_module, dense_module,
                    planned_controller=read_path if use_ps_pipeline else plugin._controller,
                )
            if use_ps_pipeline:
                read_path.shutdown()
            if plugin is not None:
                shutdown_start = time.perf_counter()
                plugin.shutdown()
                lifecycle["bagpipe_shutdown_ms"] = (time.perf_counter() - shutdown_start) * 1e3
                plugin = None
            if use_bagpipe or use_ps_pipeline or dense_graph is not None:
                lifecycle_path = Path(cfg.output_root) / "outputs" / cfg.run_id / f"lifecycle_rank{rank}.json"
                lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
                lifecycle_path.write_text(json.dumps(lifecycle, indent=2) + "\n")
            if cfg.save_checkpoint:
                from ..checkpoint import export_checkpoint
                export_checkpoint(
                    Path(cfg.checkpoint_path),
                    cfg=cfg, step=cfg.steps,
                    dense_module=unwrapped_module,
                    embedding_module=embedding_module,
                    dense_optimizer=dense_optimizer,
                    sparse_optimizer=sparse_optimizer,
                    rank=rank,
                )
            if use_dist and dist.is_initialized():
                dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
                dist.destroy_process_group()
            return {
                "backend": "recstore",
                "rows": rows,
            }
        except BaseException as error:
            continuous_error = error if continuous is not None else None
            if "read_path" in locals() and getattr(read_path, "metadata_lookahead", 0):
                try:
                    read_path._metadata_failed(error)
                except BaseException as cleanup_error:
                    if continuous is None:
                        raise
                    error.add_note("Metadata failure propagation also failed: " + repr(cleanup_error))
            raise
        finally:
            if continuous_error is not None:
                # Keep the first training error and attempt every owned cleanup.
                cleanups = []
                if step_iter is not None:
                    cleanups.append(step_iter.close)
                if events is not None:
                    cleanups.append(events.close)
                if plugin is not None:
                    cleanups.append(plugin.shutdown)
                if "read_path" in locals():
                    cleanups.append(lambda: read_path.shutdown(flush=False))
                cleanups.append(lambda: continuous.drain_after_error(continuous_error))
                cleanups.append(lambda: os.chdir(str(orig_cwd)))
                for cleanup in cleanups:
                    try:
                        cleanup()
                    except BaseException as cleanup_error:
                        continuous_error.add_note("Continuous worker cleanup also failed: " + repr(cleanup_error))
            else:
                if step_iter is not None:
                    step_iter.close()
                if events is not None:
                    events.close()
                if plugin is not None:
                    if use_bagpipe:
                        plugin._controller.shutdown(flush=False)
                    else:
                        plugin.shutdown()
                if cfg.read_mode == "ps_pipeline" and "read_path" in locals():
                    read_path.shutdown(flush=False)
                os.chdir(str(orig_cwd))

    def run(self, repo_root: Path, cfg: RunConfig) -> dict:
        if cfg.backend != "recstore":
            raise ValueError("RecStoreRunner requires cfg.backend to be 'recstore'.")
        validate_recstore_config(cfg)

        worker = _read_worker_context(
            "recstore", default_world_size=cfg.nnodes * cfg.nproc_per_node
        )
        if worker is not None:
            ensure_shared_dir(worker.output_dir)
            return self._run_local_worker(
                repo_root=repo_root, cfg=cfg, rank=worker.rank,
                world_size=worker.world_size, local_rank=worker.local_rank,
                out_csv=worker.output_dir / f"rank{worker.rank}.csv",
            )

        if cfg.nnodes * cfg.nproc_per_node <= 1:
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
