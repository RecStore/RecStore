from __future__ import annotations

import os
from typing import Sequence

import torch


def sync_device(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_layer_sizes(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("layer size string must not be empty")
    return [int(part) for part in values]


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
    if os.environ.get("RECSTORE_ORDERED_KT_VIEW") == "1" and all(
        callable(getattr(embeddings, name, None))
        for name in ("keys", "values", "key_dim", "length_per_key")
    ):
        names = list(feature_names)
        values = embeddings.values()
        widths = embeddings.length_per_key()
        if (names and len(set(names)) == len(names)
                and list(embeddings.keys()) == names and embeddings.key_dim() == 1
                and len(widths) == len(names) and widths[0] >= 0
                and all(width == widths[0] for width in widths)
                and isinstance(values, torch.Tensor) and not values.requires_grad
                and values.device.type in ("cpu", "cuda") and values.ndim == 2
                and values.is_contiguous() and values.shape[1] == len(names) * widths[0]):
            # No-grad ordered slices are already laid out as [B, F, D].
            return values.view(values.shape[0], len(names), widths[0])
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
    synchronize: bool = True,
):
    if synchronize:
        sync_device(torch, device)
    dense_features = dense_batch.to(device, non_blocking=True)
    embedded_sparse = embedded_sparse_source.to(device, non_blocking=True)
    if detach_sparse:
        embedded_sparse = embedded_sparse.detach().requires_grad_(True)
    labels = labels_batch.to(device, non_blocking=True).float()
    if labels.ndim == 1:
        labels = labels.view(-1, 1)
    if synchronize:
        sync_device(torch, device)
    return dense_features, embedded_sparse, labels


def run_hybrid_backward(loss, embedded_sparse, dense_module, torch, device, *, synchronize=True):
    if synchronize:
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
    if synchronize:
        sync_device(torch, device)
    return embedded_sparse_grad.detach()
