from __future__ import annotations

import unittest

import torch


def compute_missing_ids(cur_ids: torch.Tensor, known_ids: set[int]) -> list[int]:
    return [x for x in cur_ids.tolist() if x not in known_ids]


class TestWarmupIds(unittest.TestCase):
    def test_compute_missing_ids(self) -> None:
        known = {1, 3, 5}
        cur = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        self.assertEqual(compute_missing_ids(cur, known), [2, 4, 6])

    def test_compute_missing_ids_empty(self) -> None:
        known = {1, 2}
        cur = torch.tensor([1, 2], dtype=torch.int64)
        self.assertEqual(compute_missing_ids(cur, known), [])


if __name__ == "__main__":
    unittest.main()

