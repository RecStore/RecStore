#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
# from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
# [yinj]
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
RECSTORE_PATH = os.path.join(REPO_ROOT, 'src')
DLRM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)
if DLRM_PATH not in sys.path:
    sys.path.insert(0, DLRM_PATH)

from dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from launch_config import (
    apply_launch_config,
    build_config_from_sources,
    extract_explicit_config_keys,
)
from python.pytorch.torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection

try:
    from data.custom_dataloader import get_dataloader
    from lr_scheduler import LRPolicyScheduler
    from multi_hot import Multihot, RestartableMap
except ImportError:
    try:
        from ..data.custom_dataloader import get_dataloader
        from ..lr_scheduler import LRPolicyScheduler
        from ..multi_hot import Multihot, RestartableMap
    except ImportError:
        print("Warning: Could not import custom dataloader modules")
        def get_dataloader(args, backend, stage):
            raise NotImplementedError("Please ensure custom dataloader modules are available")

try:
    from report_uploader import report_metric
except ImportError:
    print("Warning: Could not import report_uploader. Reporting disabled.")
    def report_metric(*args): return True

def report_latency(name: str, ms_value: float):
    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    
    import json
    from tools.config.recstore_config_path import resolve_recstore_config_path

    try:
        with open(resolve_recstore_config_path(), 'r') as f:
            config = json.load(f)
            value_type = (
                config.get('cache_ps', {})
                .get('base_kv_config', {})
                .get('value', {})
                .get('type', 'DRAM_VALUE_STORE')
            )
            storage_str = value_type
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        storage_str = "DRAM" 
    
    val_us = ms_value * 1000.0
    
    import time
    start_us = int(time.time() * 1e6)
    key = f"{name}_{device_str}_{storage_str}|{start_us}"
    report_metric("op_latency", key, "recstore_us", val_us)


try:
    from profiling_utils import print_env_config, CudaTimer
except ImportError:
    # Fallback if running from a different directory context
    sys.path.append(os.path.dirname(__file__))
    try:
        from profiling_utils import print_env_config, CudaTimer
    except ImportError:
        print("Warning: Could not import profiling_utils")
        def print_env_config(mode): pass
        class CudaTimer:
            def __init__(self, enable=True): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def elapsed_ms(self): return 0.0


TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.


class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer for single day data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=None,
        help="number of embeddings (hashes) per sparse feature",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="comma separated list of number of embeddings per sparse feature",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="number of embedding dimensions",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="type of interaction layer to use",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="number of DCN layers",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="low rank dimension for DCN",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="comma separated layer sizes for over arch",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="comma separated layer sizes for dense arch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory for faster GPU transfer",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="Use memory mapping for loading data",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Path to preprocessed Criteo dataset",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Path to synthetic multi-hot Criteo dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1tb",
        help="Dataset name (criteo_1tb or criteo_kaggle)",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle batches",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle training set",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Validation frequency within epoch",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Use Adagrad optimizer",
    )
    parser.add_argument(
        "--single_day_mode",
        dest="single_day_mode",
        action="store_true",
        help="Enable single day data training mode",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (rest for validation) in single day mode",
    )
    parser.add_argument(
        "--random-dataset",
        dest="random_dataset",
        action="store_true",
        help="Use generated random data instead of processed day_0 files.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=4194304,
        help="Synthetic dataset size used when --random-dataset is enabled.",
    )
    parser.add_argument(
        "--enable_prefetch",
        action="store_true",
        help="Enable async embedding prefetch to overlap I/O and compute",
    )
    parser.add_argument(
        "--prefetch_depth",
        type=int,
        default=2,
        help="Prefetch queue depth (batches)",
    )
    parser.add_argument(
        "--fuse-emb-tables",
        dest="fuse_emb_tables",
        action="store_true",
        help="Enable embedding table fusion (single pull + prefix encoding).",
    )
    parser.add_argument(
        "--no-fuse-emb-tables",
        dest="fuse_emb_tables",
        action="store_false",
        help="Disable embedding table fusion and use per-feature pulls.",
    )
    parser.set_defaults(fuse_emb_tables=True)
    parser.add_argument(
        "--fuse-k",
        type=int,
        default=30,
        help="Bit prefix shift (k) used for fusion: fused_id=(table_idx<<k)|row_id",
    )
    parser.add_argument(
        "--trace_file",
        type=str,
        default=None,
        help="Profiler chrome trace file path",
    )
    parser.add_argument(
        "--ps-host",
        type=str,
        default=None,
        help="PS server host",
    )
    parser.add_argument(
        "--ps-port",
        type=int,
        default=None,
        help="PS server port",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--gin_config",
        type=str,
        default=None,
        help="Path to gin launch config file.",
    )
    parser.add_argument(
        "--gin_binding",
        dest="gin_bindings",
        action="append",
        default=[],
        help="Gin binding override. Can be passed multiple times.",
    )
    
    args = parser.parse_args(argv)
    explicit_config_keys = extract_explicit_config_keys(argv)
    launch_config = build_config_from_sources(
        gin_config=args.gin_config,
        gin_bindings=args.gin_bindings,
        cli_overrides={},
    )
    apply_launch_config(args, launch_config, explicit_config_keys)

    if args.single_day_mode:
        if not args.random_dataset and args.in_memory_binary_criteo_path is None:
            raise ValueError("--in_memory_binary_criteo_path must be specified for single day mode")
        
        if args.num_embeddings_per_feature is None:
            # Random-data smoke runs should fit on a single 24GB GPU without
            # allocating production-scale embedding tables by default.
            LIMIT_FEATURE = 100000 if args.random_dataset else 800000
            orig_list = [
                40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
                40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976,
                14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36,
            ]
            clamped = [str(min(v, LIMIT_FEATURE)) for v in orig_list]
            args.num_embeddings_per_feature = ",".join(clamped)
        
        if not args.adagrad:
            args.learning_rate = 0.005
            args.adagrad = True
        
        if args.random_dataset:
            print(f"Single day mode enabled. Training with generated random data only.")
            print(f"Synthetic dataset size: {args.dataset_size}")
        else:
            print(f"Single day mode enabled. Training with day_0 data only.")
        print(f"Training ratio: {args.train_ratio}")
        print(f"Validation ratio: {1 - args.train_ratio}")
    
    return args


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        dist.init_process_group(backend="gloo")
        dist.init_process_group(backend="gloo")
        device = torch.device("cpu")
    
    print_env_config("RecStore")
    
    print(f"Distributed training initialized:")
    print(f"  Rank: {dist.get_rank()}")
    print(f"  World size: {dist.get_world_size()}")
    print(f"  Device: {device}")
    
    if hasattr(args, "allow_tf32") and args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

    train_dataloader = get_dataloader(args, "nccl" if torch.cuda.is_available() else "gloo", "train")
    val_dataloader = get_dataloader(args, "nccl" if torch.cuda.is_available() else "gloo", "val")
    
    def custom_collate(batch):
        if not batch:
            return batch

        from torchrec import KeyedJaggedTensor
        feature_names = DEFAULT_CAT_NAMES

        # CustomCriteoDataset.__getitems__ already returns batched tensors.
        if (
            isinstance(batch, tuple)
            and len(batch) == 3
            and torch.is_tensor(batch[0])
            and batch[0].dim() == 2
        ):
            dense_batch, sparse_mat, labels_batch = batch
            sparse_mat = sparse_mat.to(torch.long)
            B = sparse_mat.shape[0]
        else:
            dense_features = []
            sparse_features = []
            labels = []

            for dense, sparse, label in batch:
                dense_features.append(torch.as_tensor(dense, dtype=torch.float32))
                sparse_features.append(sparse)
                labels.append(torch.as_tensor(label, dtype=torch.float32))

            dense_batch = torch.stack(dense_features)
            labels_batch = torch.stack(labels)
            sparse_mat = torch.stack([s.to(torch.long) for s in sparse_features], dim=0)
            B = sparse_mat.shape[0]
        values_list = [sparse_mat[:, i] for i in range(26)]
        values = torch.cat(values_list, dim=0)
        one_lengths = torch.ones(B, dtype=torch.int32, device=values.device)
        lengths_list = [one_lengths for _ in range(26)]
        lengths = torch.cat(lengths_list, dim=0)
        sparse_batch = KeyedJaggedTensor.from_lengths_sync(
            keys=feature_names,
            values=values,
            lengths=lengths,
        )

        return dense_batch, sparse_batch, labels_batch
    
    train_dataloader = DataLoader(
        train_dataloader.dataset,
        batch_size=train_dataloader.batch_size,
        shuffle=True,
        drop_last=args.drop_last_training_batch,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataloader.dataset,
        batch_size=val_dataloader.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    if args.num_embeddings_per_feature is not None:
        num_embeddings_per_feature = [
            int(x) for x in args.num_embeddings_per_feature.split(",")
        ]
    else:
        num_embeddings_per_feature = [args.num_embeddings] * 26
    
    eb_configs = [
        {
            "name": f"t_{feature_name}",
            "num_embeddings": num_embeddings_per_feature[feature_idx],
            "embedding_dim": args.embedding_dim,
            "feature_names": [feature_name],
        }
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    embedding_bag_collection = RecStoreEmbeddingBagCollection(
        eb_configs,
        enable_fusion=args.fuse_emb_tables,
        fusion_k=args.fuse_k,
        ps_host=args.ps_host,
        ps_port=args.ps_port,
    )
    print(f"Embedding fusion enabled: {args.fuse_emb_tables}; fusion_k={args.fuse_k}")
    prefetch_enabled = args.enable_prefetch
    if prefetch_enabled:
        try:
            from python.pytorch.recstore.Dataset import RecStoreDataset
        except ImportError:
            from src.python.pytorch.recstore.Dataset import RecStoreDataset

        def key_extractor(batch):
            # batch is (dense, sparse, labels)
            _, sparse, _ = batch
            ids_map = {}
            for key in sparse.keys():
                ids_map[key] = sparse[key].values()
            return ids_map

        train_dataloader = RecStoreDataset(
            train_dataloader, 
            embedding_bag_collection.kv_client, 
            key_extractor=key_extractor,
            prefetch_count=args.prefetch_depth
        )
    
    if args.interaction_type == InteractionType.DCN:
        model = DLRM_DCN(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=13,
            dense_arch_layer_sizes=[int(x) for x in args.dense_arch_layer_sizes.split(",")],
            over_arch_layer_sizes=[int(x) for x in args.over_arch_layer_sizes.split(",")],
            dcn_num_layers=args.dcn_num_layers,
            dcn_low_rank_dim=args.dcn_low_rank_dim,
        )
    else:
        model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=13,
            dense_arch_layer_sizes=[int(x) for x in args.dense_arch_layer_sizes.split(",")],
            over_arch_layer_sizes=[int(x) for x in args.over_arch_layer_sizes.split(",")],
        )
    
    model = model.to(device)
    
    dense_params = []
    sparse_modules = []
    
    for name, module in model.named_modules():
        if hasattr(module, '_config_names') and hasattr(module, '_trace'):
            sparse_modules.append(module)
        elif not name.startswith(('embedding_bag_collection', 'ebc')):
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    dense_params.append(param)
    
    if not dense_params:
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith(('embedding_bag_collection', 'ebc')):
                dense_params.append(param)
    
    # Create optimizers
    from python.pytorch.recstore.optimizer import SparseSGD
    
    sparse_optimizer = SparseSGD(sparse_modules, lr=args.learning_rate)
    
    if dense_params:
        if args.adagrad:
            dense_optimizer = torch.optim.Adagrad(dense_params, lr=args.learning_rate)
        else:
            dense_optimizer = torch.optim.SGD(dense_params, lr=args.learning_rate)
    else:
        dense_optimizer = None
    
    criterion = torch.nn.BCEWithLogitsLoss()
    auroc = metrics.AUROC(task="binary")
    
    from torch.profiler import profile, ProfilerActivity, schedule
    from torch.profiler import tensorboard_trace_handler

    class _NullProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def step(self):
            pass

    profiler_cm = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=5, active=2, repeat=1),
            on_trace_ready=(lambda p: p.export_chrome_trace(args.trace_file)) if (dist.get_rank() == 0 and args.trace_file) else (tensorboard_trace_handler("./logs_recstore") if dist.get_rank() == 0 else None),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if args.trace_file
        else _NullProfiler()
    )
    with profiler_cm as prof:
        for epoch in range(args.epochs):
            # Restart prefetch iterator each epoch to avoid blocking after exhaustion
            # RecStoreDataset already starts its producer in __init__. Restarting
            # before epoch 0 kills that thread while it is still loading the first
            # shuffled batch of the full Criteo set.
            if epoch > 0 and prefetch_enabled and hasattr(train_dataloader, "restart"):
                train_dataloader.restart()
            model.train()
            print(f"Epoch {epoch + 1}/{args.epochs}")
            auroc.reset()
            
            train_loss = 0.0
            train_auroc = 0.0
            num_batches = 0
            forward_time_total = 0.0
            backward_time_total = 0.0
            opt_time_total = 0.0
            dense_opt_time_total = 0.0
            sparse_opt_time_total = 0.0
            emb_time_total = 0.0
            nn_time_total = 0.0
            # Additional profiling
            prefetch_wait_time_total = 0.0
            emb_update_time_total = 0.0

            use_cuda_timing = torch.cuda.is_available() and device.type == 'cuda'
            if use_cuda_timing:
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_end = torch.cuda.Event(enable_timing=True)
                bwd_start = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                opt_start = torch.cuda.Event(enable_timing=True)
                opt_end = torch.cuda.Event(enable_timing=True)
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                if prefetch_enabled:
                     # RecStoreDataset yields (batch_data, handles)
                     # batch_data is (dense, sparse, labels)
                     batch_data, handles = batch
                     dense_features, sparse_features, labels = batch_data
                     # deliver handles to EBC so it consumes prefetched results
                     # deliver handles to EBC so it consumes prefetched results
                     with CudaTimer(use_cuda_timing) as timer_pf:
                        embedding_bag_collection.set_prefetch_handles(handles)
                     # Note: set_prefetch_handles itself doesn't block, but accessing results later will.
                     # We can't easily measure just the "wait" time here without modifying EBC internals 
                     # or instrumenting where EBC calls wait_and_get.
                     # Assuming the EBC modification isn't done, we rely on EBC report_prefetch_stats used later.
                     # However, user asked for "each operation time".
                     # We will try to hook into the KV client pull/update if possible, or trust report_prefetch_stats.

                else:
                    dense_features, sparse_features, labels = batch
                dense_features = dense_features.to(device)
                sparse_features = sparse_features.to(device)
                labels = labels.to(device)
                
                if dense_optimizer is not None:
                    dense_optimizer.zero_grad()
                sparse_optimizer.zero_grad()
                
                if use_cuda_timing:
                    fwd_start.record()
                else:
                    t_fwd_start = time.time()
                outputs = model(dense_features, sparse_features)
                loss = criterion(outputs, labels.float())
                
                if use_cuda_timing:
                    fwd_end.record()
                else:
                    t_fwd_end = time.time()
                
                if use_cuda_timing:
                    bwd_start.record()
                else:
                    t_bwd_start = time.time()
                loss.backward()

                if use_cuda_timing:
                    bwd_end.record()
                else:
                    t_bwd_end = time.time()

                
                if use_cuda_timing:
                    opt_start.record()
                else:
                    t_opt_start = time.time()

                emb_ms = 0.0
                nn_ms = 0.0
                dense_opt_ms = 0.0
                sparse_opt_ms = 0.0
                raw_model = model.module if hasattr(model, "module") else model

                if use_cuda_timing:
                    torch.cuda.synchronize()
                    t_dense_opt_start = time.time()
                    if dense_optimizer is not None:
                        dense_optimizer.step()
                    torch.cuda.synchronize()
                    t_dense_opt_end = time.time()

                    torch.cuda.synchronize()
                    t_sparse_opt_start = time.time()
                    sparse_optimizer.step()
                    if hasattr(sparse_optimizer, "flush"):
                        sparse_optimizer.flush()
                    torch.cuda.synchronize()
                    t_sparse_opt_end = time.time()

                    opt_end.record()
                    torch.cuda.synchronize()
                    fwd_ms = fwd_start.elapsed_time(fwd_end)
                    bwd_ms = bwd_start.elapsed_time(bwd_end)
                    opt_ms = opt_start.elapsed_time(opt_end)
                    dense_opt_ms = (t_dense_opt_end - t_dense_opt_start) * 1000.0
                    sparse_opt_ms = (t_sparse_opt_end - t_sparse_opt_start) * 1000.0
                    
                    if hasattr(raw_model, "timings"):
                        events = raw_model.timings
                        emb_ms = events["emb_start"].elapsed_time(events["emb_end"])
                        dense1_ms = events["dense_start"].elapsed_time(events["dense_end"])
                        dense2_ms = events["dense2_start"].elapsed_time(events["dense2_end"])
                        nn_ms = dense1_ms + dense2_ms
                else:
                    t_dense_opt_start = time.time()
                    if dense_optimizer is not None:
                        dense_optimizer.step()
                    t_dense_opt_end = time.time()

                    t_sparse_opt_start = time.time()
                    sparse_optimizer.step()
                    if hasattr(sparse_optimizer, "flush"):
                        sparse_optimizer.flush()
                    t_sparse_opt_end = time.time()

                    t_opt_end = time.time()
                    fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0
                    bwd_ms = (t_bwd_end - t_bwd_start) * 1000.0
                    opt_ms = (t_opt_end - t_opt_start) * 1000.0
                    dense_opt_ms = (t_dense_opt_end - t_dense_opt_start) * 1000.0
                    sparse_opt_ms = (t_sparse_opt_end - t_sparse_opt_start) * 1000.0
                    
                    if hasattr(raw_model, "timings_cpu"):
                        emb_ms = raw_model.timings_cpu.get("sparse_ms", 0.0)
                        nn_ms = raw_model.timings_cpu.get("dense_ms", 0.0)
                
                forward_time_total += fwd_ms
                backward_time_total += bwd_ms
                opt_time_total += opt_ms
                dense_opt_time_total += dense_opt_ms
                sparse_opt_time_total += sparse_opt_ms
                emb_time_total += emb_ms
                nn_time_total += nn_ms
                
                # Report latencies
                report_latency("Forward", fwd_ms)
                report_latency("Backward", bwd_ms)
                report_latency("Optimizer", opt_ms)
                report_latency("DenseOptimizer", dense_opt_ms)
                report_latency("SparseOptimizer", sparse_opt_ms)
                report_latency("Embedding", emb_ms)
                report_latency("Dense", nn_ms)
                
                # Extract stats from EBC if available for this batch
                curr_pf_wait = 0.0
                if prefetch_enabled:
                    # report_prefetch_stats returns cumulative or since reset? 
                    # Code in EmbeddingBag.py says "if reset: clear()".
                    # We print stats at end of epoch with reset=True.
                    # To get per-batch, we'd need to peek or reset frequently.
                    # Let's peek into stats safely if possible
                    if hasattr(embedding_bag_collection, "_prefetch_wait_latencies") and embedding_bag_collection._prefetch_wait_latencies:
                         curr_pf_wait = embedding_bag_collection._prefetch_wait_latencies[-1] * 1000.0 # s to ms
                prefetch_wait_time_total += curr_pf_wait

                prof.step()
                
                train_loss += loss.item()
                auroc_score = auroc(outputs.detach().squeeze(), labels.squeeze())
                train_auroc += auroc_score.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    pf_msg = f" PF_Wait(ms)={curr_pf_wait:.2f}" if prefetch_enabled else ""
                    print(f"Batch {batch_idx}: Loss={loss.item():.4f} AUROC={auroc_score.item():.4f} FWD={fwd_ms:.2f} (Emb={emb_ms:.2f} NN={nn_ms:.2f}) BWD={bwd_ms:.2f} OPT={opt_ms:.2f} (DenseOPT={dense_opt_ms:.2f} SparseOPT={sparse_opt_ms:.2f}){pf_msg}")
            
            avg_train_loss = train_loss / num_batches
            avg_train_auroc = auroc.compute().item()
            
            avg_fwd = forward_time_total / num_batches if num_batches else 0.0
            avg_bwd = backward_time_total / num_batches if num_batches else 0.0
            avg_opt = opt_time_total / num_batches if num_batches else 0.0
            avg_dense_opt = dense_opt_time_total / num_batches if num_batches else 0.0
            avg_sparse_opt = sparse_opt_time_total / num_batches if num_batches else 0.0
            avg_emb = emb_time_total / num_batches if num_batches else 0.0
            avg_nn = nn_time_total / num_batches if num_batches else 0.0
            avg_pf_wait = prefetch_wait_time_total / num_batches if num_batches else 0.0
            print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}, Training AUROC: {avg_train_auroc:.4f}, AvgFWD(ms): {avg_fwd:.2f} (Emb: {avg_emb:.2f} NN: {avg_nn:.2f}), AvgBWD(ms): {avg_bwd:.2f}, AvgOPT(ms): {avg_opt:.2f} (DenseOPT: {avg_dense_opt:.2f} SparseOPT: {avg_sparse_opt:.2f}), AvgPF_Wait(ms): {avg_pf_wait:.2f}")
            
            model.eval()
            auroc.reset()
            val_loss = 0.0
            val_num_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    dense_features = batch[0].to(device)
                    sparse_features = batch[1].to(device)
                    labels = batch[2].to(device)
                    
                    outputs = model(dense_features, sparse_features)
                    loss = criterion(outputs, labels.float())
                    
                    val_loss += loss.item()
                    auroc.update(outputs.squeeze(), labels.squeeze())
                    val_num_batches += 1
            
            avg_val_loss = val_loss / val_num_batches
            avg_val_auroc = auroc.compute().item()
            
            print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}, Validation AUROC: {avg_val_auroc:.4f}")
            
            model.train()
            auroc.reset()
            if prefetch_enabled:
                stats = embedding_bag_collection.report_prefetch_stats(reset=True)
                print(f"Prefetch Stats (Epoch {epoch + 1}): batches={stats['batches_prefetched']}, avg_wait_ms={stats['avg_wait_ms']:.3f}, avg_issue_to_wait_ms={stats['avg_issue_to_wait_ms']:.3f}, total_ids={stats['total_prefetched_ids']}, embeddings_per_sec_during_wait={stats['embeddings_per_sec_during_wait']:.1f}")
    
        print("Training completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
