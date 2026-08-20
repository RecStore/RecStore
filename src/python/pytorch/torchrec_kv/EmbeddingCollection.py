import inspect
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
    from torchrec.modules.embedding_configs import EmbeddingConfig
except ImportError:
    class KeyedJaggedTensor:  # pragma: no cover - fallback typing surface
        pass

    class JaggedTensor:
        def __init__(
            self,
            *,
            values: torch.Tensor,
            lengths: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
        ) -> None:
            self._values = values
            self._lengths = lengths
            self._weights = weights

        def values(self) -> torch.Tensor:
            return self._values

        def lengths(self) -> torch.Tensor:
            return self._lengths

        def weights(self) -> Optional[torch.Tensor]:
            return self._weights

    @dataclass
    class EmbeddingConfig:
        name: str
        embedding_dim: int
        num_embeddings: int
        feature_names: List[str]

try:
    from ..recstore.KVClient import RecStoreClient, get_kv_client
except ImportError:
    from recstore.KVClient import RecStoreClient, get_kv_client


@dataclass
class _EmbeddingConfigView:
    name: str
    embedding_dim: int
    num_embeddings: int
    feature_names: List[str]
    embedding_names: List[str]


def _get_required_field(config: Any, field: str) -> Any:
    if isinstance(config, Mapping):
        return config[field]
    return getattr(config, field)


def _get_optional_field(config: Any, field: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(field, default)
    return getattr(config, field, default)


def _normalize_embedding_config(config: Any) -> _EmbeddingConfigView:
    name = str(_get_required_field(config, "name"))
    embedding_dim = int(_get_required_field(config, "embedding_dim"))
    num_embeddings = int(_get_required_field(config, "num_embeddings"))
    feature_names = list(_get_required_field(config, "feature_names"))
    embedding_names = _get_optional_field(config, "embedding_names", None)
    if embedding_names is None:
        # TorchRec returns feature names for ordinary EmbeddingConfig tables.
        embedding_names = feature_names
    embedding_names = list(embedding_names)
    if len(feature_names) != len(embedding_names):
        raise ValueError(
            f"EmbeddingConfig '{name}' must have the same number of "
            "feature_names and embedding_names."
        )
    return _EmbeddingConfigView(
        name=name,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        feature_names=feature_names,
        embedding_names=embedding_names,
    )


def _normalize_tables(tables: Any) -> List[_EmbeddingConfigView]:
    if tables is None:
        return []
    if isinstance(tables, Mapping):
        configs = list(tables.values())
    else:
        configs = list(tables)
    return [_normalize_embedding_config(config) for config in configs]


class RecStoreEmbeddingCollection(torch.nn.Module):
    """TorchRec EmbeddingCollection-compatible RecStore adapter.

    The module keeps TorchRec's unpooled output contract:
    KeyedJaggedTensor -> Dict[str, JaggedTensor]. RecStore owns embedding row
    storage; backward hooks collect sparse gradients for SparseSGD.
    """

    def __init__(
        self,
        tables: Optional[Sequence[EmbeddingConfig]] = None,
        *,
        embedding_configs: Any = None,
        need_indices: bool = False,
        lr: float = 0.01,
        enable_fusion: bool = True,
        fusion_k: int = 30,
        ps_host: Optional[str] = None,
        ps_port: Optional[int] = None,
        kv_client: Optional[RecStoreClient] = None,
        initialize_tables: bool = True,
        initialize_values: bool = False,
        init_func: Optional[Callable[[Any, torch.dtype], torch.Tensor]] = None,
        clamp_ids: bool = False,
    ) -> None:
        super().__init__()
        if tables is None:
            tables = embedding_configs
        self._embedding_configs = _normalize_tables(tables)
        if not self._embedding_configs:
            raise ValueError("RecStoreEmbeddingCollection requires at least one table.")

        self.kv_client: RecStoreClient = kv_client if kv_client is not None else get_kv_client()
        if ps_host is not None and ps_port is not None:
            self.kv_client.set_ps_config(ps_host, ps_port)

        self._need_indices = bool(need_indices)
        self._lr = lr
        self._enable_fusion = bool(enable_fusion)
        self._fusion_k = int(fusion_k)
        self._clamp_ids = bool(clamp_ids)
        self.is_recstore_sparse_module = True
        self._trace: List[Dict[str, torch.Tensor]] = []

        if not self._enable_fusion and len(self._embedding_configs) > 1:
            raise ValueError(
                "Multiple RecStore embedding tables require enable_fusion=True "
                "because backend lookups share one key space."
            )
        if self._enable_fusion and self._fusion_k < 0:
            raise ValueError("fusion_k must be non-negative.")
        if self._enable_fusion and len(self._embedding_configs) > 1:
            max_rows_per_table = 1 << self._fusion_k
            for config in self._embedding_configs:
                if config.num_embeddings > max_rows_per_table:
                    raise ValueError(
                        f"Embedding table '{config.name}' with "
                        f"{config.num_embeddings} rows does not fit in "
                        f"fusion_k={self._fusion_k}."
                    )

        self._master_config = self._embedding_configs[0]
        self.feature_keys: List[str] = []
        self._config_names: Dict[str, str] = {}
        self._feature_table_indices: Dict[str, int] = {}
        self._feature_configs: Dict[str, _EmbeddingConfigView] = {}
        self._feature_output_names: Dict[str, str] = {}
        for table_idx, config in enumerate(self._embedding_configs):
            for feature_name, embedding_name in zip(
                config.feature_names,
                config.embedding_names,
            ):
                if feature_name in self._feature_configs:
                    raise ValueError(f"Duplicate feature name: {feature_name}")
                self.feature_keys.append(feature_name)
                self._config_names[feature_name] = config.name
                self._feature_table_indices[feature_name] = table_idx
                self._feature_configs[feature_name] = config
                self._feature_output_names[feature_name] = embedding_name

        if self._enable_fusion:
            master_dim = self._master_config.embedding_dim
            for config in self._embedding_configs:
                if config.embedding_dim != master_dim:
                    raise ValueError(
                        "enable_fusion=True requires all embedding tables to "
                        "have the same embedding_dim."
                    )

        if initialize_tables:
            self._initialize_tables(
                initialize_values=initialize_values,
                init_func=init_func,
            )

    def _initialize_tables(
        self,
        *,
        initialize_values: bool,
        init_func: Optional[Callable[[Any, torch.dtype], torch.Tensor]],
    ) -> None:
        for table_idx, config in enumerate(self._embedding_configs):
            base_offset = (table_idx << self._fusion_k) if self._enable_fusion else 0
            self._initialize_table(
                config,
                base_offset=base_offset,
                initialize_values=initialize_values,
                init_func=init_func,
            )

    def _initialize_table(
        self,
        config: _EmbeddingConfigView,
        *,
        base_offset: int,
        initialize_values: bool,
        init_func: Optional[Callable[[Any, torch.dtype], torch.Tensor]],
    ) -> None:
        if self._init_data_supports_initialize_values():
            self.kv_client.init_data(
                name=config.name,
                shape=(config.num_embeddings, config.embedding_dim),
                dtype=torch.float32,
                base_offset=base_offset,
                init_func=init_func,
                initialize_values=initialize_values,
            )
            return

        if initialize_values:
            if init_func is not None:
                raise RuntimeError(
                    "Custom initialization requires kv_client.init_data(..., "
                    "init_func=..., initialize_values=True) support."
                )
            self.kv_client.init_data(
                name=config.name,
                shape=(config.num_embeddings, config.embedding_dim),
                dtype=torch.float32,
                base_offset=base_offset,
            )
            return

        register_tensor_meta = getattr(self.kv_client, "register_tensor_meta", None)
        init_embedding_table = getattr(self.kv_client, "init_embedding_table", None)
        if not callable(register_tensor_meta) or not callable(init_embedding_table):
            raise RuntimeError(
                "initialize_values=False requires kv_client.init_data(..., "
                "initialize_values=False) or register_tensor_meta plus "
                "init_embedding_table support."
            )
        register_tensor_meta(
            name=config.name,
            shape=(config.num_embeddings, config.embedding_dim),
            dtype=torch.float32,
            base_offset=base_offset,
        )
        ok = init_embedding_table(
            config.name,
            config.num_embeddings,
            config.embedding_dim,
        )
        if ok is False:
            raise RuntimeError(
                f"Failed to initialize embedding table '{config.name}' on backend."
            )

    def _init_data_supports_initialize_values(self) -> bool:
        try:
            signature = inspect.signature(self.kv_client.init_data)
        except (TypeError, ValueError):
            return True
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return "initialize_values" in signature.parameters

    def embedding_configs(self) -> List[_EmbeddingConfigView]:
        return list(self._embedding_configs)

    def recstore_checkpoint_config(self) -> Dict[str, Any]:
        tables = []
        for table_idx, config in enumerate(self._embedding_configs):
            base_offset = (
                table_idx << self._fusion_k if self._enable_fusion else 0
            )
            tables.append(
                {
                    "name": config.name,
                    "num_embeddings": config.num_embeddings,
                    "embedding_dim": config.embedding_dim,
                    "base_offset": base_offset,
                    "feature_names": list(config.feature_names),
                    "embedding_names": list(config.embedding_names),
                }
            )
        return {
            "tables": tables,
            "fusion": {
                "enabled": self._enable_fusion,
                "fusion_k": self._fusion_k,
            },
            "clamp_ids": self._clamp_ids,
            "need_indices": self._need_indices,
        }

    def reset_trace(self) -> None:
        self._trace = []

    def _append_trace(self, name: str, ids: torch.Tensor, grad: torch.Tensor) -> None:
        ids_view = ids.detach().to(device=grad.device, dtype=torch.int64)
        grad_view = grad.detach().to(torch.float32)
        if ids_view.numel() == 0:
            return
        self._trace.append(
            {
                "name": name,
                "ids": ids_view,
                "grads": grad_view,
            }
        )

    def _lookup_ids_for_feature(
        self,
        feature_name: str,
        values: torch.Tensor,
    ) -> tuple[str, str, torch.Tensor]:
        config = self._feature_configs[feature_name]
        lookup_ids = values.to(dtype=torch.int64)
        if self._clamp_ids:
            lookup_ids = torch.clamp(
                lookup_ids,
                min=0,
                max=max(int(config.num_embeddings) - 1, 0),
            )
        if self._enable_fusion:
            table_idx = self._feature_table_indices[feature_name]
            lookup_ids = lookup_ids + (table_idx << self._fusion_k)
            return self._master_config.name, self._master_config.name, lookup_ids
        return config.name, config.name, lookup_ids

    def _empty_values_like_feature(
        self,
        feature_name: str,
        values: torch.Tensor,
    ) -> torch.Tensor:
        config = self._feature_configs[feature_name]
        return torch.empty(
            (0, config.embedding_dim),
            dtype=torch.float32,
            device=values.device,
        )

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        feature_embeddings: Dict[str, JaggedTensor] = {}
        for feature_name in self.feature_keys:
            jt = features[feature_name]
            values = jt.values()
            lengths = jt.lengths()
            read_name, trace_name, lookup_ids = self._lookup_ids_for_feature(
                feature_name,
                values,
            )
            if lookup_ids.numel() == 0:
                embeddings = self._empty_values_like_feature(feature_name, values)
            else:
                embeddings = self.kv_client.pull(
                    name=read_name,
                    ids=lookup_ids,
                )
                if embeddings.device != values.device:
                    embeddings = embeddings.to(values.device)

            if torch.is_grad_enabled() and embeddings.numel() > 0:
                embeddings.requires_grad_()

                def grad_hook(
                    grad: torch.Tensor,
                    name: str = trace_name,
                    ids: torch.Tensor = lookup_ids,
                ) -> None:
                    self._append_trace(name, ids, grad)

                embeddings.register_hook(grad_hook)

            output_name = self._feature_output_names[feature_name]
            feature_embeddings[output_name] = JaggedTensor(
                values=embeddings,
                lengths=lengths,
                weights=values if self._need_indices else None,
            )
        return feature_embeddings
