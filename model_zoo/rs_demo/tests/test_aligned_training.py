from __future__ import annotations

import unittest

import torch

from model_zoo.rs_demo.runtime.aligned_training import prepare_dense_input


class _FakeCuda:
    def __init__(self) -> None:
        self.calls = 0

    def synchronize(self, device) -> None:
        self.calls += 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def cat(self, tensors, dim=0):
        return torch.cat(tensors, dim=dim)


class TestAlignedTraining(unittest.TestCase):
    def test_prepare_dense_input_builds_leaf_pooled_and_syncs(self) -> None:
        fake_torch = _FakeTorch()
        dense_batch = torch.randn(2, 13)
        pooled_cpu = torch.randn(2, 8)
        labels_batch = torch.tensor([0.0, 1.0])

        dense_input, pooled, labels = prepare_dense_input(
            dense_batch=dense_batch,
            pooled_source=pooled_cpu,
            labels_batch=labels_batch,
            torch=fake_torch,
            device=torch.device("cuda"),
        )

        self.assertEqual(tuple(dense_input.shape), (2, 21))
        self.assertTrue(pooled.requires_grad)
        self.assertIsNone(pooled.grad_fn)
        self.assertEqual(tuple(labels.shape), (2, 1))
        self.assertEqual(fake_torch.cuda.calls, 2)


if __name__ == "__main__":
    unittest.main()
