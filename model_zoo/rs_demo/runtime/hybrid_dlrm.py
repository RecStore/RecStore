from __future__ import annotations

from typing import Sequence

try:
    from ...torchrec_dlrm.dlrm import DenseArch, InteractionArch, OverArch
except ImportError:
    from torchrec_dlrm.dlrm import DenseArch, InteractionArch, OverArch


def sync_device(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_layer_sizes(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("layer size string must not be empty")
    return [int(part) for part in values]


class HybridDenseArch:
    def __init__(
        self,
        dense_in_features: int,
        embedding_dim: int,
        num_sparse_features: int,
        dense_arch_layer_sizes: Sequence[int],
        over_arch_layer_sizes: Sequence[int],
        device,
    ) -> None:
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

    def parameters(self):
        yield from self.dense_arch.parameters()
        yield from self.inter_arch.parameters()
        yield from self.over_arch.parameters()

    def __call__(self, dense_features, embedded_sparse):
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
    dense_features = dense_batch.to(device)
    embedded_sparse = embedded_sparse_source.to(device)
    if detach_sparse:
        embedded_sparse = embedded_sparse.detach().requires_grad_(True)
    labels = labels_batch.to(device).float()
    if labels.ndim == 1:
        labels = labels.view(-1, 1)
    sync_device(torch, device)
    return dense_features, embedded_sparse, labels


def run_hybrid_backward(loss, embedded_sparse, dense_module, torch, device):
    sync_device(torch, device)
    dense_params = [param for param in dense_module.parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, dense_params + [embedded_sparse], retain_graph=True)
    for param, grad in zip(dense_params, grads[:-1]):
        param.grad = None if grad is None else grad.detach()
    embedded_sparse_grad = grads[-1]
    if embedded_sparse_grad is None:
        raise RuntimeError("missing embedded_sparse gradient after backward")
    sync_device(torch, device)
    return embedded_sparse_grad.detach()
