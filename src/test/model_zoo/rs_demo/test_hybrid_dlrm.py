from __future__ import annotations

import unittest

import torch

from model_zoo.rs_demo.models.dlrm import build_hybrid_dense_arch
from model_zoo.rs_demo.models.utils import (
    flatten_embedded_sparse_grad_for_recstore,
    prepare_hybrid_dlrm_input,
    reshape_recstore_embeddings_for_dlrm,
    run_hybrid_backward,
)


class TestHybridDlrm(unittest.TestCase):
    def test_reshape_recstore_embeddings_for_dlrm_restores_bfd_layout(self) -> None:
        embeddings = torch.tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
                [60.0, 61.0],
            ]
        )

        embedded_sparse = reshape_recstore_embeddings_for_dlrm(
            embeddings=embeddings,
            batch_rows=2,
            num_sparse_features=3,
        )

        self.assertEqual(tuple(embedded_sparse.shape), (2, 3, 2))
        self.assertTrue(
            torch.equal(
                embedded_sparse,
                torch.tensor(
                    [
                        [[10.0, 11.0], [30.0, 31.0], [50.0, 51.0]],
                        [[20.0, 21.0], [40.0, 41.0], [60.0, 61.0]],
                    ]
                ),
            )
        )

    def test_flatten_embedded_sparse_grad_for_recstore_restores_feature_major_order(self) -> None:
        embedded_sparse_grad = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )

        flat = flatten_embedded_sparse_grad_for_recstore(embedded_sparse_grad)

        self.assertEqual(tuple(flat.shape), (6, 2))
        self.assertTrue(
            torch.equal(
                flat,
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [7.0, 8.0],
                        [3.0, 4.0],
                        [9.0, 10.0],
                        [5.0, 6.0],
                        [11.0, 12.0],
                    ]
                ),
            )
        )

    def test_build_hybrid_dense_arch_matches_dlrm_output_shape(self) -> None:
        model = build_hybrid_dense_arch(
            torch=torch,
            dense_in_features=13,
            embedding_dim=4,
            num_sparse_features=3,
            dense_arch_layer_sizes=[8, 4],
            over_arch_layer_sizes=[16, 1],
            device=torch.device("cpu"),
        )
        dense = torch.randn(2, 13)
        embedded_sparse = torch.randn(2, 3, 4, requires_grad=True)

        logits = model(dense, embedded_sparse)

        self.assertEqual(tuple(logits.shape), (2, 1))
        loss = logits.sum()
        dense_grads = torch.autograd.grad(loss, list(model.parameters()) + [embedded_sparse])
        self.assertEqual(len(dense_grads), len(list(model.parameters())) + 1)
        self.assertIsInstance(model, torch.nn.Module)

    def test_run_hybrid_backward_populates_parameter_grads_via_backward(self) -> None:
        model = torch.nn.Linear(4, 1)
        embedded_sparse = torch.randn(2, 4, requires_grad=True)
        loss = model(embedded_sparse).sum()

        embedded_sparse_grad = run_hybrid_backward(
            loss=loss,
            embedded_sparse=embedded_sparse,
            dense_module=model,
            torch=torch,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(embedded_sparse_grad.shape), tuple(embedded_sparse.shape))
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

    def test_split_backward_can_propagate_grad_to_non_leaf_sparse_source(self) -> None:
        model = torch.nn.Linear(4, 1)
        leaf_sparse = torch.randn(2, 4, requires_grad=True)
        embedded_sparse_source = leaf_sparse * 2.0
        dense_batch = torch.randn(2, 13)
        labels_batch = torch.ones(2, 1)

        _, embedded_sparse, _ = prepare_hybrid_dlrm_input(
            dense_batch=dense_batch,
            embedded_sparse_source=embedded_sparse_source,
            labels_batch=labels_batch,
            torch=torch,
            device=torch.device("cpu"),
            detach_sparse=True,
        )
        loss = model(embedded_sparse).sum()

        embedded_sparse_grad = run_hybrid_backward(
            loss=loss,
            embedded_sparse=embedded_sparse,
            dense_module=model,
            torch=torch,
            device=torch.device("cpu"),
        )
        embedded_sparse_source.backward(embedded_sparse_grad)

        self.assertEqual(tuple(embedded_sparse_grad.shape), tuple(embedded_sparse.shape))
        self.assertIsNotNone(leaf_sparse.grad)


if __name__ == "__main__":
    unittest.main()
