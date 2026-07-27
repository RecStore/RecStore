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
