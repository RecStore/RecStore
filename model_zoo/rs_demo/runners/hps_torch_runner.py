from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ..config import (
    RunConfig,
    ensure_shared_dir,
    resolve_num_embeddings_per_feature,
    validate_hps_torch_config,
)
from ..data.dlrm_source import (
    build_kjt_batch_from_dense_sparse_labels,
    build_train_dataloader,
    get_default_cat_names,
    inject_project_paths,
)
from ..runtime.hps_torch_embedding import (
    HpsTorchEmbeddingBagCollection,
    import_hps_torch_module,
    prepare_hps_torch_model_files,
)
from ..models.dlrm import build_hybrid_dense_arch
from ..models.utils import (
    sync_device,
    parse_layer_sizes,
    prepare_hybrid_dlrm_input,
    reshape_torchrec_embeddings_for_dlrm,
    run_hybrid_backward,
)
from python.pytorch.recstore.benchmark.report import finalize_torchrec_row, write_stage_csv
from .base import BenchmarkRunner
from ..runtime.timing import stage_timer
from .torchrec_runner import (
    _barrier_for_step_alignment,
    _merge_rank_outputs,
    _pick_socket_ifname,
)


def ensure_hps_torch_available() -> None:
    try:
        import_hps_torch_module()
    except RuntimeError as exc:
        raise RuntimeError(
            "HPS Torch backend requires `merlin-hps` / `hps_torch` to be installed."
        ) from exc


def _bool_int(flag: bool) -> int:
    return 1 if flag else 0


def _model_ready_path(cfg: RunConfig) -> Path:
    return Path(cfg.hps_torch_model_dir) / ".hps_torch_model_ready"


def _wait_for_model_files(cfg: RunConfig, timeout_s: float = 600.0) -> None:
    ready_path = _model_ready_path(cfg)
    deadline = time.monotonic() + timeout_s
    while not ready_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for HPS Torch model files: {ready_path}")
        time.sleep(0.2)


def _rank_config_path(cfg: RunConfig, rank: int) -> Path:
    base = Path(cfg.hps_torch_config_file)
    if rank == 0:
        return base
    return base.with_name(f"{base.stem}_rank{rank}{base.suffix}")


def _build_train_dataloader_for_rank(
    repo_root: Path,
    cfg: RunConfig,
    rank: int,
    world_size: int,
):
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


class HpsTorchRunner(BenchmarkRunner):
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _rank_output_dir(self, cfg: RunConfig) -> Path:
        return Path(cfg.output_root) / "outputs" / cfg.run_id / "hps_torch_ranks"

    def _build_torchrun_cmd(self, repo_root: Path, cfg: RunConfig) -> list[str]:
        rdzv_endpoint = f"{cfg.master_addr}:{cfg.master_port}"
        cmd = [
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
            "hps_torch",
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
            "--hps-torch-model-name",
            str(cfg.hps_torch_model_name),
            "--hps-torch-config-file",
            str(Path(cfg.hps_torch_config_file)),
            "--hps-torch-model-dir",
            str(Path(cfg.hps_torch_model_dir)),
            "--hps-torch-main-csv",
            str(Path(cfg.hps_torch_main_csv)),
            "--hps-torch-main-agg-csv",
            str(Path(cfg.hps_torch_main_agg_csv)),
            "--hps-torch-key-offset-mode",
            str(cfg.hps_torch_key_offset_mode),
            "--hps-torch-gpucacheper",
            str(cfg.hps_torch_gpucacheper),
            "--no-start-server",
        ]
        if cfg.num_embeddings_per_feature:
            cmd.extend(
                [
                    "--num-embeddings-per-feature",
                    str(cfg.num_embeddings_per_feature),
                ]
            )
        if not cfg.hps_torch_materialize_embeddings:
            cmd.append("--hps-torch-no-materialize-embeddings")
        if cfg.hps_torch_force_materialize:
            cmd.append("--hps-torch-force-materialize")
        if not cfg.hps_torch_gpucache:
            cmd.append("--hps-torch-disable-gpucache")
        return cmd

    def _run_worker(
        self,
        repo_root: Path,
        cfg: RunConfig,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        out_csv: Path,
    ) -> dict[str, Any]:
        inject_project_paths(repo_root)
        default_cat_names = get_default_cat_names()

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_dist = world_size > 1
        if use_dist and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.manual_seed(int(cfg.seed) + int(rank))

        dataset, dataloader = _build_train_dataloader_for_rank(
            repo_root=repo_root,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
        )
        del dataset
        num_embeddings_per_feature = resolve_num_embeddings_per_feature(
            cfg.num_embeddings,
            cfg.num_embeddings_per_feature,
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
        model_dir = Path(cfg.hps_torch_model_dir)
        config_path = _rank_config_path(cfg, rank)
        materialize = bool(cfg.hps_torch_materialize_embeddings and rank == 0)
        if rank != 0:
            _wait_for_model_files(cfg)
        table_specs = prepare_hps_torch_model_files(
            eb_configs,
            model_root=model_dir,
            config_path=config_path,
            model_name=cfg.hps_torch_model_name,
            max_batch_size=cfg.batch_size,
            device_id=local_rank,
            seed=cfg.seed,
            key_offset_mode=cfg.hps_torch_key_offset_mode,
            materialize=materialize,
            force_materialize=bool(cfg.hps_torch_force_materialize and rank == 0),
            gpucache=cfg.hps_torch_gpucache,
            gpucacheper=cfg.hps_torch_gpucacheper,
        )
        if rank == 0:
            _model_ready_path(cfg).write_text("ready\n", encoding="utf-8")

        embedding_module = HpsTorchEmbeddingBagCollection(
            eb_configs,
            ps_config_file=str(config_path),
            model_name=cfg.hps_torch_model_name,
            table_specs=table_specs,
        ).to(device)
        dense_module = build_hybrid_dense_arch(
            torch=torch,
            dense_in_features=13,
            embedding_dim=cfg.embedding_dim,
            num_sparse_features=len(default_cat_names),
            dense_arch_layer_sizes=parse_layer_sizes(cfg.dense_arch_layer_sizes),
            over_arch_layer_sizes=parse_layer_sizes(cfg.over_arch_layer_sizes),
            device=device,
        )
        if use_dist:
            dense_module = torch.nn.parallel.DistributedDataParallel(
                dense_module,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        criterion = torch.nn.BCEWithLogitsLoss()
        dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)

        rows: list[dict[str, Any]] = []
        data_iter = iter(dataloader)
        for step in range(cfg.steps):
            row: dict[str, Any] = {
                "backend": "hps_torch",
                "model_backend_label": "hps_torch:LookupLayer",
                "nproc": world_size,
                "rank": rank,
                "batch_size": cfg.batch_size,
                "step": step,
                "warmup_excluded": _bool_int(step < cfg.warmup_steps),
                "collective_mode": "not_measured_single_process"
                if world_size == 1
                else "dense_ddp_measured_embedding_local",
                "collective_measured": _bool_int(use_dist),
                "nnodes": cfg.nnodes,
                "nproc_per_node": cfg.nproc_per_node,
                "world_size": world_size,
                "dist_mode": "single_node",
                "torchrec_dist_mode": "",
                "torchrec_memory_mode": "",
                "torchrec_role": "trainer",
                "torchrec_is_trainer": 1,
                "hps_torch_model_name": cfg.hps_torch_model_name,
                "hps_torch_key_offset_mode": cfg.hps_torch_key_offset_mode,
            }
            step_start = time.perf_counter()

            with stage_timer(row, "batch_prepare_ms"):
                try:
                    dense_batch, sparse_batch, labels_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    dense_batch, sparse_batch, labels_batch = next(data_iter)

            with stage_timer(row, "input_pack_ms"):
                dense_batch, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                    dense_batch,
                    sparse_batch,
                    labels_batch,
                    device=device,
                )
                sync_device(torch, device)

            with stage_timer(row, "embed_lookup_local_ms"):
                embeddings = embedding_module(sparse_features)
                sync_device(torch, device)

            with stage_timer(row, "embed_pool_local_ms"):
                embedded_sparse_source = reshape_torchrec_embeddings_for_dlrm(
                    embeddings=embeddings,
                    feature_names=default_cat_names,
                    torch=torch,
                )
                sync_device(torch, device)

            with stage_timer(row, "output_unpack_ms"):
                dense_features, embedded_sparse, labels = prepare_hybrid_dlrm_input(
                    dense_batch=dense_batch,
                    embedded_sparse_source=embedded_sparse_source,
                    labels_batch=labels_batch,
                    torch=torch,
                    device=device,
                    detach_sparse=True,
                )

            with stage_timer(row, "dense_fwd_ms"):
                logits = dense_module(dense_features, embedded_sparse)
                loss = criterion(logits, labels)
                sync_device(torch, device)

            with stage_timer(row, "backward_ms"):
                _embedded_sparse_grad = run_hybrid_backward(
                    loss=loss,
                    embedded_sparse=embedded_sparse,
                    dense_module=dense_module,
                    torch=torch,
                    device=device,
                )

            with stage_timer(row, "dense_optimizer_ms"):
                dense_optimizer.step()
                dense_optimizer.zero_grad(set_to_none=True)
                sync_device(torch, device)

            row["sparse_optimizer_ms"] = 0.0
            row["collective_launch_ms"] = 0.0
            row["collective_wait_ms"] = 0.0
            row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
            rows.append(finalize_torchrec_row(row))
            _barrier_for_step_alignment(
                dist=torch.distributed,
                device=device,
                local_rank=local_rank,
                use_dist=use_dist,
            )
            if (step + 1) % 10 == 0:
                print(
                    f"[rs_demo] hps_torch step {step + 1}/{cfg.steps} "
                    f"emb={rows[-1]['emb_stage_ms']:.2f}ms "
                    f"step={rows[-1]['step_total_ms']:.2f}ms"
                )

        out_path = Path(cfg.hps_torch_main_csv)
        if out_csv != out_path:
            out_path = out_csv
        write_stage_csv(out_path, rows)
        print(f"[rs_demo] hps_torch main csv: {out_path}")
        if use_dist and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return {"backend": "hps_torch", "rows": rows}

    def _run_single_process(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        return self._run_worker(
            repo_root=repo_root,
            cfg=cfg,
            rank=0,
            world_size=1,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            out_csv=Path(cfg.hps_torch_main_csv),
        )

    def _run_distributed(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rank_dir = self._rank_output_dir(cfg)
        ensure_shared_dir(rank_dir)
        for path in rank_dir.glob("rank*.csv"):
            path.unlink()
        ready_path = _model_ready_path(cfg)
        try:
            ready_path.unlink()
        except FileNotFoundError:
            pass

        cmd = self._build_torchrun_cmd(repo_root, cfg)
        env = os.environ.copy()
        env["RS_DEMO_HPS_TORCH_WORKER"] = "1"
        env["RS_DEMO_HPS_TORCH_WORKER_DIR"] = str(rank_dir)
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
                "hps_torch torchrun worker failed\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}"
            )

        rank_csvs = [rank_dir / f"rank{rank}.csv" for rank in range(cfg.nproc_per_node)]
        missing = [str(path) for path in rank_csvs if not path.exists()]
        if missing:
            raise RuntimeError(f"missing HPS Torch rank csv outputs: {missing}")
        rows = _merge_rank_outputs(rank_csvs, Path(cfg.hps_torch_main_csv))
        return {"backend": "hps_torch", "rows": rows}

    def run(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        if cfg.backend != "hps_torch":
            raise ValueError("HpsTorchRunner requires cfg.backend to be 'hps_torch'.")
        validate_hps_torch_config(cfg)
        if not torch.cuda.is_available():
            raise RuntimeError("hps_torch.LookupLayer requires CUDA.")
        if os.environ.get("RS_DEMO_HPS_TORCH_WORKER") == "1":
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", str(cfg.nproc_per_node)))
            worker_dir = Path(os.environ["RS_DEMO_HPS_TORCH_WORKER_DIR"])
            ensure_shared_dir(worker_dir)
            out_csv = worker_dir / f"rank{rank}.csv"
            return self._run_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                out_csv=out_csv,
            )
        if cfg.nproc_per_node <= 1:
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
