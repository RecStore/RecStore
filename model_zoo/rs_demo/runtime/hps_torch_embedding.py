from __future__ import annotations

import json
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from types import ModuleType
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
except ModuleNotFoundError:  # pragma: no cover - import-time fallback for light tests
    @dataclass
    class EmbeddingBagConfig:
        name: str
        embedding_dim: int
        num_embeddings: int
        feature_names: list[str]

    class KeyedJaggedTensor:
        pass

    class KeyedTensor:
        def __init__(self, *, keys: list[str], values: torch.Tensor, length_per_key: list[int]) -> None:
            self._keys = list(keys)
            self._values = values
            self._length_per_key = list(length_per_key)
            offsets = [0]
            for width in self._length_per_key:
                offsets.append(offsets[-1] + int(width))
            self._offsets = dict(zip(self._keys, zip(offsets[:-1], offsets[1:])))

        def keys(self) -> list[str]:
            return list(self._keys)

        def values(self) -> torch.Tensor:
            return self._values

        def length_per_key(self) -> list[int]:
            return list(self._length_per_key)

        def __getitem__(self, key: str) -> torch.Tensor:
            start, end = self._offsets[key]
            return self._values[:, start:end]


@dataclass(frozen=True)
class HpsTableSpec:
    name: str
    feature_name: str
    num_embeddings: int
    embedding_dim: int
    key_offset: int
    sparse_file: str


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]


def _local_hps_torch_source_path(repo_root: Path | None = None) -> Path:
    root = _repo_root_from_this_file() if repo_root is None else Path(repo_root)
    return root / "third_party" / "HugeCTR" / "hps_torch"


def import_hps_torch_module(repo_root: Path | None = None) -> ModuleType:
    try:
        return importlib.import_module("hps_torch")
    except ModuleNotFoundError as exc:
        if exc.name != "hps_torch":
            raise RuntimeError("hps_torch import failed because a dependency is missing.") from exc
        local_source = _local_hps_torch_source_path(repo_root)
        if (local_source / "hps_torch").is_dir():
            local_source_str = str(local_source)
            if local_source_str not in sys.path:
                sys.path.insert(0, local_source_str)
            try:
                return importlib.import_module("hps_torch")
            except ModuleNotFoundError as local_exc:
                if local_exc.name != "hps_torch":
                    raise RuntimeError(
                        "hps_torch import failed because a dependency is missing."
                    ) from local_exc
            except OSError as local_exc:
                raise RuntimeError(
                    "hps_torch was found, but its native library failed to load. "
                    "Rebuild or reinstall the HPS Torch plugin."
                ) from local_exc
        raise RuntimeError(
            "hps_torch backend requires the merlin-hps package or the local "
            "third_party/HugeCTR/hps_torch package to be available."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "hps_torch was found, but its native library failed to load. "
            "Rebuild or reinstall the HPS Torch plugin."
        ) from exc


def _normalize_configs(configs: Sequence[Any]) -> list[EmbeddingBagConfig]:
    normalized: list[EmbeddingBagConfig] = []
    for cfg in configs:
        if isinstance(cfg, EmbeddingBagConfig):
            normalized.append(cfg)
        elif isinstance(cfg, dict):
            normalized.append(EmbeddingBagConfig(**cfg))
        else:
            raise TypeError(f"unsupported embedding config type: {type(cfg)!r}")
    if not normalized:
        raise ValueError("HPS Torch embedding requires at least one table")
    return normalized


def build_hps_table_specs(
    configs: Sequence[Any],
    model_dir: Path,
    *,
    key_offset_mode: str = "cumulative",
) -> list[HpsTableSpec]:
    normalized = _normalize_configs(configs)
    if key_offset_mode not in {"none", "cumulative"}:
        raise ValueError("key_offset_mode must be 'none' or 'cumulative'")

    specs: list[HpsTableSpec] = []
    next_offset = 0
    for cfg in normalized:
        if len(cfg.feature_names) != 1:
            raise ValueError("HPS Torch adapter expects one feature per table")
        offset = 0 if key_offset_mode == "none" else next_offset
        sparse_file = model_dir / cfg.name
        specs.append(
            HpsTableSpec(
                name=str(cfg.name),
                feature_name=str(cfg.feature_names[0]),
                num_embeddings=int(cfg.num_embeddings),
                embedding_dim=int(cfg.embedding_dim),
                key_offset=int(offset),
                sparse_file=str(sparse_file),
            )
        )
        next_offset += int(cfg.num_embeddings)
    return specs


def materialize_hps_embedding_tables(
    specs: Sequence[HpsTableSpec],
    *,
    seed: int,
    chunk_rows: int = 65536,
    force: bool = False,
) -> None:
    for table_idx, spec in enumerate(specs):
        table_dir = Path(spec.sparse_file)
        table_dir.mkdir(parents=True, exist_ok=True)
        key_path = table_dir / "key"
        vec_path = table_dir / "emb_vector"
        meta_path = table_dir / "recstore_hps_meta.json"
        expected_meta = {
            "name": spec.name,
            "feature_name": spec.feature_name,
            "num_embeddings": spec.num_embeddings,
            "embedding_dim": spec.embedding_dim,
            "key_offset": spec.key_offset,
            "seed": int(seed) + table_idx,
        }
        if not force and key_path.exists() and vec_path.exists() and meta_path.exists():
            try:
                current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current_meta = {}
            if current_meta == expected_meta:
                continue

        rng = np.random.default_rng(int(seed) + table_idx)
        with key_path.open("wb") as key_file, vec_path.open("wb") as vec_file:
            for start in range(0, spec.num_embeddings, int(chunk_rows)):
                end = min(start + int(chunk_rows), spec.num_embeddings)
                keys = np.arange(
                    spec.key_offset + start,
                    spec.key_offset + end,
                    dtype=np.int64,
                )
                vectors = rng.random(
                    (end - start, spec.embedding_dim),
                    dtype=np.float32,
                )
                keys.tofile(key_file)
                vectors.tofile(vec_file)
        meta_path.write_text(json.dumps(expected_meta, sort_keys=True, indent=2), encoding="utf-8")


def write_hps_torch_config(
    config_path: Path,
    specs: Sequence[HpsTableSpec],
    *,
    model_name: str,
    max_batch_size: int,
    device_id: int,
    fuse_embedding_table: bool = True,
    gpucache: bool = True,
    gpucacheper: float = 1.0,
    max_query_per_table_per_sample: int = 1,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "supportlonglong": True,
        "fuse_embedding_table": bool(fuse_embedding_table),
        "models": [
            {
                "model": str(model_name),
                "sparse_files": [str(spec.sparse_file) for spec in specs],
                "num_of_worker_buffer_in_pool": len(specs),
                "embedding_table_names": [str(spec.name) for spec in specs],
                "embedding_vecsize_per_table": [int(spec.embedding_dim) for spec in specs],
                "maxnum_catfeature_query_per_table_per_sample": [
                    int(max_query_per_table_per_sample) for _ in specs
                ],
                "default_value_for_each_table": [0.0 for _ in specs],
                "deployed_device_list": [int(device_id)],
                "max_batch_size": int(max_batch_size),
                "cache_refresh_percentage_per_iteration": 1.0,
                "hit_rate_threshold": 1.0,
                "gpucacheper": float(gpucacheper),
                "gpucache": bool(gpucache),
                "embedding_cache_type": "static",
                "use_context_stream": True,
            }
        ],
    }
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def prepare_hps_torch_model_files(
    configs: Sequence[Any],
    *,
    model_root: Path,
    config_path: Path,
    model_name: str,
    max_batch_size: int,
    device_id: int,
    seed: int,
    key_offset_mode: str = "cumulative",
    fuse_embedding_table: bool = True,
    materialize: bool = True,
    force_materialize: bool = False,
    gpucache: bool = True,
    gpucacheper: float = 1.0,
) -> list[HpsTableSpec]:
    specs = build_hps_table_specs(
        configs,
        model_root,
        key_offset_mode=key_offset_mode,
    )
    if materialize:
        materialize_hps_embedding_tables(
            specs,
            seed=seed,
            force=force_materialize,
        )
    write_hps_torch_config(
        config_path,
        specs,
        model_name=model_name,
        max_batch_size=max_batch_size,
        device_id=device_id,
        fuse_embedding_table=fuse_embedding_table,
        gpucache=gpucache,
        gpucacheper=gpucacheper,
    )
    return specs


class HpsTorchEmbeddingBagCollection(torch.nn.Module):
    """TorchRec-like pooled embedding adapter backed by hps_torch.LookupLayer."""

    def __init__(
        self,
        embedding_bag_configs: Sequence[Any],
        *,
        ps_config_file: str,
        model_name: str,
        table_specs: Sequence[HpsTableSpec] | None = None,
    ) -> None:
        super().__init__()
        self._embedding_bag_configs = _normalize_configs(embedding_bag_configs)
        if table_specs is None:
            table_specs = [
                HpsTableSpec(
                    name=str(cfg.name),
                    feature_name=str(cfg.feature_names[0]),
                    num_embeddings=int(cfg.num_embeddings),
                    embedding_dim=int(cfg.embedding_dim),
                    key_offset=0,
                    sparse_file="",
                )
                for cfg in self._embedding_bag_configs
            ]
        if len(table_specs) != len(self._embedding_bag_configs):
            raise ValueError("HPS Torch adapter expects one table spec per embedding table")
        self._spec_by_feature = {spec.feature_name: spec for spec in table_specs}

        hps_torch = import_hps_torch_module()

        self._layers = torch.nn.ModuleDict()
        for table_id, cfg in enumerate(self._embedding_bag_configs):
            if len(cfg.feature_names) != 1:
                raise ValueError("HPS Torch adapter expects one feature per table")
            self._layers[cfg.feature_names[0]] = hps_torch.LookupLayer(
                ps_config_file=str(ps_config_file),
                model_name=str(model_name),
                table_id=int(table_id),
                emb_vec_size=int(cfg.embedding_dim),
            )
        self._lookup_executor = ThreadPoolExecutor(max_workers=len(self._layers))

    def close(self) -> None:
        """Shut down the background lookup thread pool."""
        self._lookup_executor.shutdown(wait=False)

    def embedding_bag_configs(self):
        return self._embedding_bag_configs

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        keys = list(features.keys())
        lookup_jobs: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []
        pooled_by_key: list[torch.Tensor | None] = []
        length_per_key: list[int] = []
        for key in keys:
            if key not in self._layers:
                raise KeyError(f"HPS Torch adapter received unknown feature: {key}")
            feature = features[key]
            values = feature.values().to(dtype=torch.int64)
            lengths = feature.lengths().to(dtype=torch.int64)
            config = next(
                (cfg for cfg in self._embedding_bag_configs if key in cfg.feature_names),
                None,
            )
            if config is None:
                raise KeyError(f"HPS Torch adapter has no embedding config for feature: {key}")
            length_per_key.append(int(config.embedding_dim))

            batch_size = int(lengths.numel())
            if int(lengths.sum().item()) != int(values.numel()):
                raise ValueError(
                    f"HPS Torch feature {key} has mismatched lengths and values: "
                    f"sum(lengths)={int(lengths.sum().item())} values={int(values.numel())}"
                )
            if values.device.type != "cuda":
                raise RuntimeError("hps_torch.LookupLayer requires CUDA sparse ids")
            if batch_size == 0:
                pooled_by_key.append(
                    torch.empty(
                        (0, int(config.embedding_dim)),
                        dtype=torch.float32,
                        device=values.device,
                    )
                )
                continue

            max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
            if max_len == 0:
                pooled_by_key.append(
                    torch.zeros(
                        (batch_size, int(config.embedding_dim)),
                        dtype=torch.float32,
                        device=values.device,
                    )
                )
                continue

            if torch.all(lengths == 1):
                query = values.reshape(batch_size, 1)
                mask = None
            else:
                query = torch.zeros((batch_size, max_len), dtype=torch.int64, device=values.device)
                mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=values.device)
                offsets = torch.cat(
                    [
                        torch.zeros((1,), dtype=torch.int64, device=values.device),
                        torch.cumsum(lengths, dim=0),
                    ]
                )
                for row in range(batch_size):
                    start = int(offsets[row].item())
                    end = int(offsets[row + 1].item())
                    row_len = end - start
                    if row_len > 0:
                        query[row, :row_len] = values[start:end]
                        mask[row, :row_len] = 1.0

            spec = self._spec_by_feature.get(key)
            if spec is not None and spec.key_offset:
                query = query + int(spec.key_offset)
            if not query.is_contiguous():
                query = query.contiguous()
            lookup_jobs.append((key, query, mask))
            pooled_by_key.append(None)

        def run_lookup(job: tuple[str, torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
            key, query, mask = job
            torch.cuda.set_device(query.device)
            looked_up = self._layers[key](query)
            if mask is not None:
                looked_up = looked_up * mask.unsqueeze(-1)
            return looked_up.sum(dim=1)

        if lookup_jobs:
            lookup_results = list(self._lookup_executor.map(run_lookup, lookup_jobs))
            result_idx = 0
            for idx, item in enumerate(pooled_by_key):
                if item is None:
                    pooled_by_key[idx] = lookup_results[result_idx]
                    result_idx += 1

        resolved = [item for item in pooled_by_key if item is not None]
        values_out = torch.cat(resolved, dim=1) if resolved else torch.empty((0, 0))
        return KeyedTensor(keys=keys, values=values_out, length_per_key=length_per_key)
