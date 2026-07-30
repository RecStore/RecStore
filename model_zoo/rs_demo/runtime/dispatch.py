from __future__ import annotations

import torch

from ..models.dlrm import build_hybrid_dense_arch


def build_dense_module(
    model_type: str,
    torch,
    dense_in_features: int,
    embedding_dim: int,
    num_sparse_features: int,
    dense_arch_layer_sizes,
    over_arch_layer_sizes,
    device,
    **_extra,
):
    """Build the dense compute module.

    Currently only the DLRM HybridDenseArch is shipped in the public repo.
    Additional model types may be available locally.
    """
    module = build_hybrid_dense_arch(
        torch=torch,
        dense_in_features=dense_in_features,
        embedding_dim=embedding_dim,
        num_sparse_features=num_sparse_features,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        device=device,
    )
    module.model_type = "dlrm"
    return module


def build_criterion(model_type: str, task_names: list[str] | None = None):
    """Loss for the selected model type."""
    return torch.nn.BCEWithLogitsLoss()


def compute_dense_loss(
    model_type: str,
    dense_module,
    criterion,
    dense_features: torch.Tensor,
    embedded_sparse: torch.Tensor,
    labels: torch.Tensor,
):
    """Forward + loss for the selected model type.

    Returns (loss, logits).
    """
    logits = dense_module(dense_features, embedded_sparse)
    loss = criterion(logits, labels)
    return loss, logits


def model_task_names(dense_module) -> list[str]:
    """Return task names for multi-task models (empty for DLRM)."""
    return []
