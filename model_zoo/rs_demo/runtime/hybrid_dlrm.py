from __future__ import annotations

from typing import Sequence

import torch

try:
    from ...torchrec_dlrm.dlrm import DenseArch, InteractionArch, OverArch
except ImportError:
    try:
        from torchrec_dlrm.dlrm import DenseArch, InteractionArch, OverArch
    except ImportError:
        class DenseArch(torch.nn.Module):
            def __init__(self, in_features: int, layer_sizes: list[int], device) -> None:
                super().__init__()
                layers: list[torch.nn.Module] = []
                current_in = int(in_features)
                for idx, out_features in enumerate(layer_sizes):
                    layers.append(torch.nn.Linear(current_in, int(out_features), device=device))
                    if idx != len(layer_sizes) - 1:
                        layers.append(torch.nn.ReLU())
                    current_in = int(out_features)
                self.model = torch.nn.Sequential(*layers)

            def forward(self, dense_features):
                return self.model(dense_features)

        class InteractionArch(torch.nn.Module):
            def __init__(self, num_sparse_features: int) -> None:
                super().__init__()
                self.num_sparse_features = int(num_sparse_features)

            def forward(self, embedded_dense, embedded_sparse):
                features = torch.cat([embedded_dense.unsqueeze(1), embedded_sparse], dim=1)
                interactions: list[torch.Tensor] = []
                num_features = features.shape[1]
                for left in range(num_features):
                    for right in range(left + 1, num_features):
                        interactions.append(
                            (features[:, left, :] * features[:, right, :]).sum(dim=1, keepdim=True)
                        )
                if interactions:
                    pairwise = torch.cat(interactions, dim=1)
                else:
                    pairwise = torch.empty(
                        (embedded_dense.shape[0], 0),
                        dtype=embedded_dense.dtype,
                        device=embedded_dense.device,
                    )
                return torch.cat([embedded_dense, pairwise], dim=1)

        class OverArch(torch.nn.Module):
            def __init__(self, in_features: int, layer_sizes: list[int], device) -> None:
                super().__init__()
                layers: list[torch.nn.Module] = []
                current_in = int(in_features)
                for idx, out_features in enumerate(layer_sizes):
                    layers.append(torch.nn.Linear(current_in, int(out_features), device=device))
                    if idx != len(layer_sizes) - 1:
                        layers.append(torch.nn.ReLU())
                    current_in = int(out_features)
                self.model = torch.nn.Sequential(*layers)

            def forward(self, interacted_features):
                return self.model(interacted_features)


def sync_device(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_layer_sizes(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("layer size string must not be empty")
    return [int(part) for part in values]


class HybridDenseArch(torch.nn.Module):
    def __init__(
        self,
        dense_in_features: int,
        embedding_dim: int,
        num_sparse_features: int,
        dense_arch_layer_sizes: Sequence[int],
        over_arch_layer_sizes: Sequence[int],
        device,
    ) -> None:
        super().__init__()
        if not dense_arch_layer_sizes:
            raise ValueError("dense_arch_layer_sizes must not be empty")
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                "dense arch final size must match embedding_dim for DLRM interaction"
            )

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=list(dense_arch_layer_sizes),
            device=device,
        )
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)
        self.over_arch = OverArch(
            in_features=embedding_dim + (num_sparse_features * (num_sparse_features + 1)) // 2,
            layer_sizes=list(over_arch_layer_sizes),
            device=device,
        )

    def to(self, device):
        self.dense_arch = self.dense_arch.to(device)
        self.inter_arch = self.inter_arch.to(device)
        self.over_arch = self.over_arch.to(device)
        return self

    def forward(self, dense_features, embedded_sparse):
        embedded_dense = self.dense_arch(dense_features)
        interacted = self.inter_arch(embedded_dense, embedded_sparse)
        return self.over_arch(interacted)


def build_hybrid_dense_arch(
    torch,
    dense_in_features: int,
    embedding_dim: int,
    num_sparse_features: int,
    dense_arch_layer_sizes: Sequence[int],
    over_arch_layer_sizes: Sequence[int],
    device,
):
    del torch
    return HybridDenseArch(
        dense_in_features=dense_in_features,
        embedding_dim=embedding_dim,
        num_sparse_features=num_sparse_features,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        device=device,
    ).to(device)


def reshape_recstore_embeddings_for_dlrm(
    embeddings,
    batch_rows: int,
    num_sparse_features: int,
):
    if batch_rows <= 0:
        raise ValueError("batch_rows must be greater than 0")
    if embeddings.shape[0] != batch_rows * num_sparse_features:
        raise ValueError("embedding rows do not match batch_rows * num_sparse_features")
    return embeddings.reshape(num_sparse_features, batch_rows, -1).permute(1, 0, 2).contiguous()


def reshape_torchrec_embeddings_for_dlrm(embeddings, feature_names: Sequence[str], torch):
    return torch.stack([embeddings[name] for name in feature_names], dim=1)


def flatten_embedded_sparse_grad_for_recstore(embedded_sparse_grad):
    return embedded_sparse_grad.permute(1, 0, 2).reshape(-1, embedded_sparse_grad.shape[-1]).contiguous()


def prepare_hybrid_dlrm_input(
    dense_batch,
    embedded_sparse_source,
    labels_batch,
    torch,
    device,
    *,
    detach_sparse: bool,
):
    sync_device(torch, device)
    dense_features = dense_batch.to(device, non_blocking=True)
    embedded_sparse = embedded_sparse_source.to(device, non_blocking=True)
    if detach_sparse:
        embedded_sparse = embedded_sparse.detach().requires_grad_(True)
    labels = labels_batch.to(device, non_blocking=True).float()
    if labels.ndim == 1:
        labels = labels.view(-1, 1)
    sync_device(torch, device)
    return dense_features, embedded_sparse, labels


def run_hybrid_backward(loss, embedded_sparse, dense_module, torch, device):
    sync_device(torch, device)
    dense_params = [param for param in dense_module.parameters() if param.requires_grad]
    for param in dense_params:
        param.grad = None
    if not embedded_sparse.is_leaf:
        embedded_sparse.retain_grad()
    embedded_sparse.grad = None
    loss.backward()
    embedded_sparse_grad = embedded_sparse.grad
    if embedded_sparse_grad is None:
        raise RuntimeError("missing embedded_sparse gradient after backward")
    sync_device(torch, device)
    return embedded_sparse_grad.detach()


# --------------------------------------------------------------------------
# Model plugin seam
#
# The shared harness (recstore/torchrec runners) is model-agnostic: it resolves
# a plugin by ``cfg.model`` and calls the four methods below.  DLRM is built in
# here; other models (e.g. RankMixer) live in their own ``model_zoo/<Model>/``
# package and are imported lazily by ``resolve_model_plugin``.
# --------------------------------------------------------------------------


class DlrmModelPlugin:
    """Default DLRM model plugin (dense arch + BCE loss)."""

    #: CLI dests this plugin owns (routed into ``cfg.model_args``); DLRM has none.
    ARG_DESTS: tuple[str, ...] = ()

    def add_arguments(self, parser) -> None:  # noqa: D401 - no extra CLI args
        return None

    def build_dense_module(self, cfg, *, num_sparse_features, embedding_dim, device):
        module = build_hybrid_dense_arch(
            torch=torch,
            dense_in_features=13,
            embedding_dim=embedding_dim,
            num_sparse_features=num_sparse_features,
            dense_arch_layer_sizes=parse_layer_sizes(cfg.dense_arch_layer_sizes),
            over_arch_layer_sizes=parse_layer_sizes(cfg.over_arch_layer_sizes),
            device=device,
        )
        module.model_type = "dlrm"
        return module

    def build_criterion(self, cfg, dense_module):
        return torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, dense_module, criterion, dense_features, embedded_sparse, labels):
        logits = dense_module(dense_features, embedded_sparse)
        return criterion(logits, labels), logits


DLRM_PLUGIN = DlrmModelPlugin()

#: model name -> import path of a module exposing ``PLUGIN`` (torch imported lazily).
_MODEL_PLUGIN_MODULES = {"rankmixer": "RankMixer.plugin"}
_PLUGIN_CACHE: dict[str, object] = {"dlrm": DLRM_PLUGIN}


def resolve_model_plugin(name: str):
    """Return the model plugin for ``name`` ("dlrm" built in, others imported).

    Tolerates both the ``model_zoo`` on-path layout (production) and the
    repo-root layout (tests).
    """
    name = name or "dlrm"
    if name not in _PLUGIN_CACHE:
        module_path = _MODEL_PLUGIN_MODULES.get(name)
        if module_path is None:
            raise ValueError(f"unknown model: {name}")
        import importlib

        module = None
        for candidate in (module_path, f"model_zoo.{module_path}"):
            try:
                module = importlib.import_module(candidate)
                break
            except ImportError:
                continue
        if module is None:
            raise ImportError(f"cannot import model plugin for {name!r} ({module_path})")
        _PLUGIN_CACHE[name] = module.PLUGIN
    return _PLUGIN_CACHE[name]


# Thin model-agnostic dispatchers used by the runners.  They resolve the plugin
# from ``cfg.model`` and forward — the runners never branch on the model.

def build_dense_module(cfg, *, num_sparse_features, embedding_dim, device):
    return resolve_model_plugin(cfg.model).build_dense_module(
        cfg, num_sparse_features=num_sparse_features,
        embedding_dim=embedding_dim, device=device,
    )


def build_criterion(cfg, dense_module):
    return resolve_model_plugin(cfg.model).build_criterion(cfg, dense_module)


def compute_dense_loss(cfg, dense_module, criterion, dense_features, embedded_sparse, labels):
    return resolve_model_plugin(cfg.model).compute_loss(
        dense_module, criterion, dense_features, embedded_sparse, labels,
    )


def model_task_names(cfg, dense_module) -> list[str]:
    return resolve_model_plugin(cfg.model).task_names(dense_module)
