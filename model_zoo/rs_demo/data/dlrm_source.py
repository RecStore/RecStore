from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from ..config import resolve_num_embeddings_per_feature


def inject_project_paths(repo_root: Path) -> None:
    recstore_src = str(repo_root / "src")
    dlrm_root = str(repo_root / "model_zoo/torchrec_dlrm")
    py_client = str(repo_root / "src/test/framework/pytorch")
    # Force these entries to the front of sys.path even when they are already
    # present (e.g. via PYTHONPATH).  This matters because torchrun launches
    # each worker as `python model_zoo/rs_demo/run_mock_stress.py`, which puts
    # model_zoo/rs_demo/ at sys.path[0]; its sibling `data` package would
    # otherwise shadow model_zoo/torchrec_dlrm/data (which provides
    # data.custom_dataloader).
    for p in (recstore_src, dlrm_root, py_client):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def build_kjt_batch_from_dense_sparse_labels(
    dense_batch: torch.Tensor,
    sparse_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    *,
    device: torch.device | None = None,
):
    del labels_batch
    cat_names = get_default_cat_names()

    sparse_mat = sparse_batch.to(torch.long)
    if device is not None and sparse_mat.device != device:
        sparse_mat = sparse_mat.to(device)
    batch_size = sparse_mat.shape[0]
    values_list = [sparse_mat[:, i] for i in range(26)]
    values = torch.cat(values_list, dim=0)
    one_lengths = torch.ones(batch_size, dtype=torch.int32, device=values.device)
    lengths = torch.cat([one_lengths for _ in range(26)], dim=0)

    kjt = build_sparse_features(cat_names, values, lengths)
    return dense_batch, kjt


def get_default_cat_names() -> list[str]:
    try:
        criteo = importlib.import_module("torchrec.datasets.criteo")
        return list(criteo.DEFAULT_CAT_NAMES)
    except ModuleNotFoundError:
        return [f"cat_{idx}" for idx in range(26)]


class _SimpleJaggedFeature:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor) -> None:
        self._values = values
        self._lengths = lengths

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


class _SimpleKeyedJaggedTensor:
    def __init__(self, keys: list[str], values: torch.Tensor, lengths: torch.Tensor) -> None:
        self._keys = list(keys)
        self._values = values
        self._lengths = lengths
        self._mapping = self._build_mapping()

    def _build_mapping(self) -> dict[str, _SimpleJaggedFeature]:
        mapping: dict[str, _SimpleJaggedFeature] = {}
        batch_size = self._lengths.shape[0] // len(self._keys)
        value_offset = 0
        for key_idx, key in enumerate(self._keys):
            lengths_chunk = self._lengths[
                key_idx * batch_size : (key_idx + 1) * batch_size
            ].contiguous()
            value_count = int(lengths_chunk.sum().item())
            values_chunk = self._values[value_offset : value_offset + value_count].contiguous()
            value_offset += value_count
            mapping[key] = _SimpleJaggedFeature(values_chunk, lengths_chunk)
        return mapping

    def keys(self) -> list[str]:
        return list(self._keys)

    def device(self) -> torch.device:
        return self._values.device

    def __getitem__(self, key: str) -> _SimpleJaggedFeature:
        return self._mapping[key]


def build_sparse_features(keys: list[str], values: torch.Tensor, lengths: torch.Tensor):
    try:
        jagged_tensor = importlib.import_module("torchrec.sparse.jagged_tensor")
        return jagged_tensor.KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=values,
            lengths=lengths,
        )
    except ModuleNotFoundError:
        return _SimpleKeyedJaggedTensor(keys, values, lengths)


def build_batch_ids_from_kjt(sparse_features) -> torch.Tensor:
    ids_chunks = []
    for key in sparse_features.keys():
        ids_chunks.append(sparse_features[key].values().to(torch.int64))
    if not ids_chunks:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(ids_chunks, dim=0).cpu().contiguous()


def build_table_offsets_from_eb_configs(eb_configs: list[dict], fusion_k: int) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for idx, cfg in enumerate(eb_configs):
        offsets[cfg["feature_names"][0]] = idx << fusion_k
    return offsets


def convert_kjt_ids_to_fused_ids(sparse_features, table_offsets: dict[str, int]) -> torch.Tensor:
    ids_chunks = []
    for key in sparse_features.keys():
        vals = sparse_features[key].values().to(torch.int64).cpu()
        ids_chunks.append(vals + table_offsets[key])
    if not ids_chunks:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(ids_chunks, dim=0).contiguous()


def build_train_dataloader(
    repo_root: Path,
    data_dir_rel: str,
    train_ratio: float,
    num_embeddings: int,
    batch_size: int,
    *,
    num_embeddings_per_feature: list[int] | str | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    rank: int | None = None,
    world_size: int | None = None,
):
    from data.custom_dataloader import CustomCriteoDataset  # type: ignore

    data_dir = (repo_root / data_dir_rel).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {data_dir}")

    nep = resolve_num_embeddings_per_feature(
        int(num_embeddings),
        num_embeddings_per_feature,
    )
    dataset = CustomCriteoDataset(
        data_dir=str(data_dir),
        stage="train",
        train_ratio=train_ratio,
        num_embeddings_per_feature=nep,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    sampler = None
    effective_shuffle = shuffle
    if world_size is not None and int(world_size) > 1:
        if rank is None:
            raise ValueError("rank must be provided when world_size > 1")
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=shuffle,
            seed=0 if seed is None else int(seed),
            drop_last=False,
        )
        effective_shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        generator=generator,
    )
    return dataset, dataloader
