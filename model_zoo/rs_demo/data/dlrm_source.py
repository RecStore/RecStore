from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def inject_project_paths(repo_root: Path) -> None:
    recstore_src = str(repo_root / "src")
    dlrm_root = str(repo_root / "model_zoo/torchrec_dlrm")
    py_client = str(repo_root / "src/framework/pytorch/python_client")
    for p in (recstore_src, dlrm_root, py_client):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_kjt_batch_from_dense_sparse_labels(
    dense_batch: torch.Tensor,
    sparse_batch: torch.Tensor,
    labels_batch: torch.Tensor,
):
    del labels_batch
    cat_names = get_default_cat_names()

    sparse_mat = sparse_batch.to(torch.long)
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


class _SimpleJaggedValues:
    def __init__(self, values: torch.Tensor) -> None:
        self._values = values

    def values(self) -> torch.Tensor:
        return self._values


class _SimpleKeyedJaggedTensor:
    def __init__(self, keys: list[str], values: torch.Tensor, lengths: torch.Tensor) -> None:
        self._keys = list(keys)
        self._values = values
        self._lengths = lengths
        self._mapping = self._build_mapping()

    def _build_mapping(self) -> dict[str, _SimpleJaggedValues]:
        mapping: dict[str, _SimpleJaggedValues] = {}
        batch_size = self._lengths.shape[0] // len(self._keys)
        for key_idx, key in enumerate(self._keys):
            chunk = self._values[key_idx * batch_size : (key_idx + 1) * batch_size].contiguous()
            mapping[key] = _SimpleJaggedValues(chunk)
        return mapping

    def keys(self) -> list[str]:
        return list(self._keys)

    def __getitem__(self, key: str) -> _SimpleJaggedValues:
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
):
    from data.custom_dataloader import CustomCriteoDataset  # type: ignore

    data_dir = (repo_root / data_dir_rel).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {data_dir}")

    nep = [int(num_embeddings)] * 26
    dataset = CustomCriteoDataset(
        data_dir=str(data_dir),
        stage="train",
        train_ratio=train_ratio,
        num_embeddings_per_feature=nep,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
    )
    return dataset, dataloader
