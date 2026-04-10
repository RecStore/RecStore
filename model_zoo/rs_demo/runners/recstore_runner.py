from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ..config import RunConfig, ensure_shared_dir, validate_recstore_config
from ..data.dlrm_source import (
    build_kjt_batch_from_dense_sparse_labels,
    build_table_offsets_from_eb_configs,
    build_train_dataloader,
    convert_kjt_ids_to_fused_ids,
    get_default_cat_names,
    inject_project_paths,
)
from ..runtime.hybrid_dlrm import (
    build_hybrid_dense_arch,
    flatten_embedded_sparse_grad_for_recstore,
    parse_layer_sizes,
    prepare_hybrid_dlrm_input,
    reshape_recstore_embeddings_for_dlrm,
    run_hybrid_backward,
    sync_device,
)
from ..runtime.recstore_distributed import ShardedRecstoreClient
from ..runtime.report import finalize_recstore_row, summarize_us, write_stage_csv
from .base import BenchmarkRunner


@contextmanager
def stage_timer(row: dict[str, Any], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        row[key] = (time.perf_counter() - start) * 1e3


def _bool_int(flag: bool) -> int:
    return 1 if flag else 0


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_stage_csv(path, rows)


def _pick_socket_ifname() -> str | None:
    preferred = ("eno1", "eno8303")
    try:
        available = set(os.listdir("/sys/class/net"))
    except OSError:
        return None
    for name in preferred:
        if name in available:
            return name
    return None


def _debug_log_path(cfg: RunConfig, rank: int) -> Path:
    return Path(cfg.output_root) / "outputs" / cfg.run_id / f"recstore_worker_rank{rank}.log"


def _append_worker_debug(cfg: RunConfig, rank: int, message: str) -> None:
    debug_path = _debug_log_path(cfg, rank)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} rank={rank} {message}\n")


def _build_worker_fingerprint(repo_root: Path) -> dict[str, dict[str, str]]:
    rel_paths = [
        "model_zoo/rs_demo/config.py",
        "model_zoo/rs_demo/runners/recstore_runner.py",
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
    if fingerprint_path.exists():
        all_fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
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


def _merge_rank_outputs(paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in paths:
        for row in _load_rows(path):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = ""
                    continue
                if key in {"backend", "dist_mode"}:
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
    merged.sort(key=lambda row: (int(row.get("rank", 0)), int(row.get("step", 0))))
    _write_rows(out_path, merged)
    return merged


def _barrier_for_step_alignment(dist, device, local_rank: int, use_dist: bool) -> None:
    if not use_dist:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _known_ids_path(cfg: RunConfig) -> Path:
    return Path(cfg.output_root) / "outputs" / cfg.run_id / "recstore_known_fused_ids.json"


def _write_known_ids_snapshot(path: Path, known_ids: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(int(key) for key in known_ids)),
        encoding="utf-8",
    )


def _load_known_ids_snapshot(path: Path) -> set[int]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"invalid known ids snapshot: {path}")
    return {int(item) for item in data}


def detect_library_path(repo_root: Path, user_path: str) -> Path:
    if user_path:
        p = Path(user_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"library_path not found: {p}")
        return p

    candidates = [
        repo_root / "build/lib/lib_recstore_ops.so",
        repo_root / "build/lib/_recstore_ops.so",
        repo_root / "build/_recstore_ops.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find RecStore ops library. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


class RecStoreRunner(BenchmarkRunner):
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _rank_output_dir(self, cfg: RunConfig) -> Path:
        return Path(cfg.output_root) / "outputs" / cfg.run_id / "recstore_ranks"

    def _build_torchrun_cmd(self, repo_root: Path, cfg: RunConfig) -> list[str]:
        rdzv_endpoint = f"{cfg.master_addr}:{cfg.master_port}"
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(cfg.nnodes),
            "--node_rank",
            str(cfg.node_rank),
            "--nproc_per_node",
            str(cfg.nproc_per_node),
            "--rdzv_backend",
            str(cfg.rdzv_backend),
            "--rdzv_endpoint",
            rdzv_endpoint,
            "--rdzv_id",
            str(cfg.rdzv_id),
            "--tee",
            "3",
            str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
            "--backend",
            "recstore",
            "--nnodes",
            str(cfg.nnodes),
            "--node-rank",
            str(cfg.node_rank),
            "--nproc-per-node",
            str(cfg.nproc_per_node),
            "--master-addr",
            str(cfg.master_addr),
            "--master-port",
            str(cfg.master_port),
            "--rdzv-backend",
            str(cfg.rdzv_backend),
            "--rdzv-id",
            str(cfg.rdzv_id),
            "--run-id",
            str(cfg.run_id),
            "--output-root",
            str(cfg.output_root),
            "--steps",
            str(cfg.steps),
            "--warmup-steps",
            str(cfg.warmup_steps),
            "--batch-size",
            str(cfg.batch_size),
            "--num-embeddings",
            str(cfg.num_embeddings),
            "--embedding-dim",
            str(cfg.embedding_dim),
            "--fuse-k",
            str(cfg.fuse_k),
            "--dense-arch-layer-sizes",
            str(cfg.dense_arch_layer_sizes),
            "--over-arch-layer-sizes",
            str(cfg.over_arch_layer_sizes),
            "--seed",
            str(cfg.seed),
            "--data-dir",
            cfg.data_dir,
            "--train-ratio",
            str(cfg.train_ratio),
            "--recstore-main-csv",
            str(Path(cfg.recstore_main_csv)),
            "--recstore-main-agg-csv",
            str(Path(cfg.recstore_main_agg_csv)),
            "--library-path",
            str(cfg.library_path),
            "--read-mode",
            str(cfg.read_mode),
            "--no-start-server",
        ]

    def _run_single_process(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        return self._run_local_worker(
            repo_root=repo_root,
            cfg=cfg,
            rank=0,
            world_size=1,
            local_rank=0,
            out_csv=Path(cfg.recstore_main_csv),
        )

    def _run_distributed(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rank_dir = self._rank_output_dir(cfg)
        ensure_shared_dir(rank_dir)

        cmd = self._build_torchrun_cmd(repo_root, cfg)
        env = os.environ.copy()
        env["RS_DEMO_RECSTORE_WORKER"] = "1"
        env["RS_DEMO_RECSTORE_WORKER_DIR"] = str(rank_dir)
        socket_ifname = _pick_socket_ifname()
        if socket_ifname:
            env.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
            env.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
        env.setdefault("NCCL_DEBUG", "WARN")
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
                "recstore torchrun worker failed\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}"
            )

        world_size = cfg.nnodes * cfg.nproc_per_node
        rank_csvs = [rank_dir / f"rank{rank}.csv" for rank in range(world_size)]
        missing = [str(path) for path in rank_csvs if not path.exists()]
        if missing:
            raise RuntimeError(f"missing rank csv outputs: {missing}")
        rows = _merge_rank_outputs(rank_csvs, Path(cfg.recstore_main_csv))
        return {"backend": "recstore", "rows": rows}

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
        from client import RecstoreClient  # type: ignore
        from torch import distributed as dist
        default_cat_names = get_default_cat_names()

        library_path = detect_library_path(repo_root, cfg.library_path)
        print(f"[rs_demo] repo_root={repo_root}")
        print(f"[rs_demo] backend=recstore")
        print(f"[rs_demo] library={library_path}")

        orig_cwd = Path.cwd()
        try:
            os.chdir(str(self.runtime_dir))
            torch.manual_seed(cfg.seed)
            use_dist = world_size > 1
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            _append_worker_debug(
                cfg,
                rank,
                f"worker_start world_size={world_size} local_rank={local_rank} backend={backend}",
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            if use_dist and not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    device_id=device if device.type == "cuda" else None,
                )
            if use_dist:
                fingerprint_path = (
                    Path(cfg.output_root)
                    / "outputs"
                    / cfg.run_id
                    / "recstore_worker_fingerprints.json"
                )
                fingerprint = _build_worker_fingerprint(repo_root)
                _write_or_verify_worker_fingerprint(
                    rank=rank,
                    world_size=world_size,
                    fingerprint=fingerprint,
                    fingerprint_path=fingerprint_path,
                )
                _append_worker_debug(cfg, rank, f"worker_fingerprint {fingerprint}")
            raw_client = RecstoreClient(library_path=str(library_path))
            client = ShardedRecstoreClient(raw_client, self.runtime_dir)
            if cfg.read_before_update and cfg.read_mode == "prefetch":
                print("[rs_demo] sharded recstore path uses prefetch read mode")
            elif cfg.read_mode != "direct":
                print("[rs_demo] unknown read mode, fallback to direct read mode")

            dataset, dataloader = build_train_dataloader(
                repo_root=repo_root,
                data_dir_rel=cfg.data_dir,
                train_ratio=cfg.train_ratio,
                num_embeddings=cfg.num_embeddings,
                batch_size=cfg.batch_size,
            )

            eb_configs = [
                {
                    "name": f"t_{feature_name}",
                    "num_embeddings": int(cfg.num_embeddings),
                    "embedding_dim": int(cfg.embedding_dim),
                    "feature_names": [feature_name],
                }
                for feature_name in default_cat_names
            ]

            if rank == 0:
                t0 = time.perf_counter()
                for table_cfg in eb_configs:
                    ok = client.init_embedding_table(
                        table_cfg["name"],
                        table_cfg["num_embeddings"],
                        table_cfg["embedding_dim"],
                    )
                    if not ok:
                        raise RuntimeError(f"init_embedding_table failed: {table_cfg['name']}")
                print(
                    f"[rs_demo] init {len(eb_configs)} tables done in {(time.perf_counter() - t0):.3f}s"
                )

            table_offsets = build_table_offsets_from_eb_configs(eb_configs, cfg.fuse_k)
            init_rows = min(int(cfg.init_rows), len(dataset))
            init_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=min(cfg.batch_size, init_rows),
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                num_workers=0,
            )

            known_fused_ids: set[int] = set()
            known_ids_path = _known_ids_path(cfg)
            if rank == 0:
                init_written = 0
                t0 = time.perf_counter()
                for dense_batch, sparse_batch, labels_batch in init_loader:
                    _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                        dense_batch, sparse_batch, labels_batch
                    )
                    fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
                    if fused_ids.numel() == 0:
                        continue
                    vals = (
                        torch.randn(fused_ids.numel(), cfg.embedding_dim, dtype=torch.float32) * 0.01
                    )
                    client.emb_write(fused_ids, vals)
                    known_fused_ids.update(fused_ids.tolist())
                    init_written += fused_ids.numel()
                    if init_written >= init_rows * 26:
                        break
                _write_known_ids_snapshot(known_ids_path, known_fused_ids)
                print(
                    f"[rs_demo] initial emb_write fused_rows={init_written} in {(time.perf_counter() - t0):.3f}s"
                )
            _barrier_for_step_alignment(
                dist=dist,
                device=device,
                local_rank=local_rank,
                use_dist=use_dist,
            )
            if use_dist and rank != 0:
                known_fused_ids = _load_known_ids_snapshot(known_ids_path)
            dense_module = build_hybrid_dense_arch(
                torch=torch,
                dense_in_features=13,
                embedding_dim=cfg.embedding_dim,
                num_sparse_features=len(default_cat_names),
                dense_arch_layer_sizes=parse_layer_sizes(cfg.dense_arch_layer_sizes),
                over_arch_layer_sizes=parse_layer_sizes(cfg.over_arch_layer_sizes),
                device=device,
            )
            criterion = torch.nn.BCEWithLogitsLoss()
            dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)

            read_lat_us: list[float] = []
            update_lat_us: list[float] = []
            rows: list[dict[str, Any]] = []
            data_iter = iter(dataloader)
            for step in range(cfg.steps):
                row: dict[str, Any] = {
                    "backend": "recstore",
                    "nproc": world_size,
                    "rank": rank,
                    "batch_size": cfg.batch_size,
                    "step": step,
                    "warmup_excluded": _bool_int(step < cfg.warmup_steps),
                    "nnodes": cfg.nnodes,
                    "nproc_per_node": cfg.nproc_per_node,
                    "world_size": cfg.nnodes * cfg.nproc_per_node,
                    "dist_mode": "multi_node" if cfg.nnodes > 1 else "single_node",
                }
                step_start = time.perf_counter()
                try:
                    raw_dense_batch, sparse_batch, labels_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    raw_dense_batch, sparse_batch, labels_batch = next(data_iter)

                with stage_timer(row, "batch_prepare_ms"):
                    dense_batch = raw_dense_batch

                with stage_timer(row, "input_pack_ms"):
                    _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                        dense_batch, sparse_batch, labels_batch
                    )
                    fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
                if fused_ids.numel() == 0:
                    continue

                read_ids = fused_ids.contiguous()
                batch_rows = dense_batch.shape[0]

                # Ensure read-before-update does not hit missing keys and crash.
                if cfg.read_before_update:
                    cur_ids = read_ids.tolist()
                    missing_ids = [x for x in cur_ids if x not in known_fused_ids]
                    if missing_ids:
                        warm_ids = torch.tensor(missing_ids, dtype=torch.int64)
                        warm_vals = torch.zeros(len(missing_ids), cfg.embedding_dim, dtype=torch.float32)
                        client.emb_write(warm_ids, warm_vals)
                        known_fused_ids.update(missing_ids)

                embeddings = None
                with stage_timer(row, "embed_lookup_local_ms"):
                    if cfg.read_before_update and cfg.read_mode == "prefetch":
                        embeddings = client.emb_read_prefetch(read_ids, cfg.embedding_dim)
                    else:
                        embeddings = client.emb_read(read_ids, cfg.embedding_dim)
                if embeddings is None:
                    raise RuntimeError("recstore emb_read returned no embeddings")
                if step >= cfg.warmup_steps:
                    read_lat_us.append(row["embed_lookup_local_ms"] * 1e3)

                with stage_timer(row, "embed_pool_local_ms"):
                    embedded_sparse_cpu = reshape_recstore_embeddings_for_dlrm(
                        embeddings=embeddings,
                        batch_rows=batch_rows,
                        num_sparse_features=len(default_cat_names),
                    )

                with stage_timer(row, "output_unpack_ms"):
                    dense_features, embedded_sparse, labels = prepare_hybrid_dlrm_input(
                        dense_batch=dense_batch,
                        embedded_sparse_source=embedded_sparse_cpu,
                        labels_batch=labels_batch,
                        torch=torch,
                        device=device,
                        detach_sparse=True,
                    )

                with stage_timer(row, "dense_fwd_ms"):
                    sync_device(torch, device)
                    logits = dense_module(dense_features, embedded_sparse)
                    loss = criterion(logits, labels)
                    sync_device(torch, device)

                with stage_timer(row, "backward_ms"):
                    embedded_sparse_grad = run_hybrid_backward(
                        loss=loss,
                        embedded_sparse=embedded_sparse,
                        dense_module=dense_module,
                        torch=torch,
                        device=device,
                    )

                with stage_timer(row, "optimizer_ms"):
                    sync_device(torch, device)
                    dense_optimizer.step()
                    dense_optimizer.zero_grad(set_to_none=True)
                    sync_device(torch, device)

                with stage_timer(row, "sparse_update_ms"):
                    sync_device(torch, device)
                    update_grads = flatten_embedded_sparse_grad_for_recstore(
                        embedded_sparse_grad.to("cpu")
                    )
                    client.emb_update_table("t_cat_0", read_ids, update_grads)
                    sync_device(torch, device)

                if step >= cfg.warmup_steps:
                    update_lat_us.append(row["sparse_update_ms"] * 1e3)
                known_fused_ids.update(read_ids.tolist())
                row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
                rows.append(finalize_recstore_row(row))
                _barrier_for_step_alignment(
                    dist=dist,
                    device=device,
                    local_rank=local_rank,
                    use_dist=use_dist,
                )

                if (step + 1) % 10 == 0:
                    print(
                        f"[rs_demo] step {step + 1}/{cfg.steps} "
                        f"emb={rows[-1]['emb_stage_ms'] if rows else 0:.2f}ms "
                        f"step={rows[-1]['step_total_ms'] if rows else 0:.2f}ms"
                    )

            print("[rs_demo] workload finished")
            print(f"[rs_demo] emb_read latency: {summarize_us(read_lat_us)}")
            print(f"[rs_demo] emb_update latency: {summarize_us(update_lat_us)}")
            _write_rows(out_csv, rows)
            if use_dist and dist.is_initialized():
                dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
                dist.destroy_process_group()
            return {
                "backend": "recstore",
                "read_lat_us": read_lat_us,
                "update_lat_us": update_lat_us,
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
            out_csv = worker_dir / f"rank{rank}.csv"
            return self._run_local_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                out_csv=out_csv,
            )

        if cfg.nnodes * cfg.nproc_per_node <= 1:
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
