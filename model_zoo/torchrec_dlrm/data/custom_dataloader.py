#!/usr/bin/env python3

import argparse
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import distributed as dist
from typing import List, Tuple

class CustomCriteoDataset(Dataset):
    def __init__(self, data_dir: str, stage: str = "train", train_ratio: float = 0.8, num_embeddings_per_feature: List[int] | None = None):
        self.data_dir = data_dir
        self.stage = stage
        self.train_ratio = train_ratio
        self.num_embeddings_per_feature = num_embeddings_per_feature
        
        self.dense_data = np.load(os.path.join(data_dir, "day_0_dense.npy"), mmap_mode="c")
        self.sparse_data = np.load(os.path.join(data_dir, "day_0_sparse.npy"), mmap_mode="c")
        self.labels_data = np.load(os.path.join(data_dir, "day_0_labels.npy"), mmap_mode="c")
        
        self.num_samples = len(self.dense_data)
        print(f"Loaded {self.num_samples} samples from day_0")
        print(f"Dense shape: {self.dense_data.shape}")
        print(f"Sparse shape: {self.sparse_data.shape}")
        print(f"Labels shape: {self.labels_data.shape}")
        
        if stage == "train":
            start_idx = 0
            end_idx = int(self.num_samples * train_ratio)
        else:
            start_idx = int(self.num_samples * train_ratio)
            end_idx = self.num_samples
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.actual_samples = end_idx - start_idx
        
        print(f"{stage.capitalize()} set: samples {start_idx} to {end_idx} ({self.actual_samples} total)")
    
    def __len__(self):
        return self.actual_samples
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx

        dense = torch.from_numpy(self.dense_data[actual_idx]).to(torch.float32)
        sparse = torch.from_numpy(self.sparse_data[actual_idx]).to(torch.int64)
        label = torch.tensor(self.labels_data[actual_idx], dtype=torch.float32)

        if self.num_embeddings_per_feature is not None and len(self.num_embeddings_per_feature) == 26:
            sparse = torch.abs(sparse)
            for i in range(26):
                vocab = int(self.num_embeddings_per_feature[i])
                if vocab > 0:
                    sparse[i] = sparse[i] % vocab
        
        return dense, sparse, label

    def __getitems__(self, indices):
        if len(indices) == 0:
            return []

        actual_indices = self.start_idx + np.asarray(indices, dtype=np.int64)
        dense_batch = torch.from_numpy(self.dense_data[actual_indices]).to(torch.float32)
        sparse_np = self.sparse_data[actual_indices].astype(np.int64, copy=False)
        sparse_batch = torch.from_numpy(sparse_np).to(torch.int64)
        labels_batch = torch.from_numpy(self.labels_data[actual_indices]).to(torch.float32)

        if self.num_embeddings_per_feature is not None and len(self.num_embeddings_per_feature) == 26:
            sparse_batch = torch.abs(sparse_batch)
            vocab = torch.tensor(self.num_embeddings_per_feature, dtype=torch.int64)
            positive = vocab > 0
            if torch.any(positive):
                sparse_batch[:, positive] %= vocab[positive]

        return dense_batch, sparse_batch, labels_batch


class RandomSingleDayDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        stage: str = "train",
        train_ratio: float = 0.8,
        num_embeddings_per_feature: List[int] | None = None,
        seed: int | None = None,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than 0")

        self.num_samples = int(num_samples)
        self.stage = stage
        self.train_ratio = train_ratio
        self.num_embeddings_per_feature = num_embeddings_per_feature or [100000] * 26
        self.seed = 0 if seed is None else int(seed)

        if stage == "train":
            start_idx = 0
            end_idx = int(self.num_samples * train_ratio)
        else:
            start_idx = int(self.num_samples * train_ratio)
            end_idx = self.num_samples

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.actual_samples = end_idx - start_idx

        print(f"Using random single-day dataset with {self.num_samples} samples")
        print(f"{stage.capitalize()} set: samples {start_idx} to {end_idx} ({self.actual_samples} total)")

    def __len__(self):
        return self.actual_samples

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        generator = torch.Generator()
        generator.manual_seed(self.seed + actual_idx)

        dense = torch.rand(13, generator=generator, dtype=torch.float32)
        sparse = torch.tensor(
            [
                torch.randint(
                    0,
                    max(int(vocab), 1),
                    (1,),
                    generator=generator,
                    dtype=torch.int64,
                ).item()
                for vocab in self.num_embeddings_per_feature
            ],
            dtype=torch.int64,
        )
        label = torch.randint(
            0,
            2,
            (1,),
            generator=generator,
            dtype=torch.int64,
        ).to(torch.float32)
        return dense, sparse, label

def get_custom_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    if hasattr(args, 'single_day_mode') and args.single_day_mode:
        if getattr(args, 'num_embeddings_per_feature', None) is not None:
            nep = [int(x) for x in args.num_embeddings_per_feature.split(",")]
        else:
            nep = [int(args.num_embeddings)] * 26 if getattr(args, 'num_embeddings', None) is not None else None

        train_ratio = args.train_ratio if hasattr(args, 'train_ratio') else 0.8
        if getattr(args, "random_dataset", False):
            dataset = RandomSingleDayDataset(
                num_samples=int(getattr(args, "dataset_size", 4194304)),
                stage=stage,
                train_ratio=train_ratio,
                num_embeddings_per_feature=nep,
                seed=getattr(args, "seed", None),
            )
        else:
            dataset = CustomCriteoDataset(
                data_dir=args.in_memory_binary_criteo_path,
                stage=stage,
                train_ratio=train_ratio,
                num_embeddings_per_feature=nep,
            )
        
        if stage in ["val", "test"] and args.test_batch_size is not None:
            batch_size = args.test_batch_size
        else:
            batch_size = args.batch_size
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(stage == "train"),
            drop_last=(stage == "train" and args.drop_last_training_batch),
            pin_memory=args.pin_memory,
            num_workers=0,  # 避免多进程问题
            collate_fn=lambda batch: batch,
        )
        
        return dataloader
    else:
        raise NotImplementedError("Custom dataloader only supports single day mode")

def get_dataloader(args: argparse.Namespace, backend: str, stage: str):
    return get_custom_dataloader(args, backend, stage) 
