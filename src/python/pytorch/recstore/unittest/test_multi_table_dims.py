from __future__ import annotations

import os
import time
import unittest

import torch

from tools.config.recstore_config_path import resolve_recstore_config_path

from ..KVClient import RecStoreClient, _KEY_TAG_SHIFT


class TestMultiTableDims(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = resolve_recstore_config_path()
        if not cls.config_path.exists():
            raise unittest.SkipTest(f"missing config file: {cls.config_path}")

    def setUp(self) -> None:
        os.environ["RECSTORE_CONFIG"] = str(self.config_path)
        RecStoreClient._instance = None
        self.client = RecStoreClient()
        self.client.set_ps_backend("hierkv")
        suffix = time.time_ns()
        self.user_table = f"user_table_{suffix}"
        self.item_table = f"item_table_{suffix}"

    def tearDown(self) -> None:
        RecStoreClient._instance = None

    def test_two_tables_different_dims_are_isolated(self) -> None:
        self.client.init_data(
            name=self.user_table, shape=(8, 4), dtype=torch.float32
        )
        self.client.init_data(
            name=self.item_table, shape=(8, 8), dtype=torch.float32
        )

        self.assertEqual(self.client._tensor_meta[self.user_table]["tag"], 0)
        self.assertEqual(self.client._tensor_meta[self.item_table]["tag"], 1)

        user_ids = torch.tensor([1, 3], dtype=torch.int64)
        item_ids = torch.tensor([1, 3], dtype=torch.int64)
        user_before = self.client.pull(self.user_table, user_ids)
        item_before = self.client.pull(self.item_table, item_ids)
        self.assertEqual(tuple(user_before.shape), (2, 4))
        self.assertEqual(tuple(item_before.shape), (2, 8))
        self.assertTrue(torch.equal(user_before, torch.zeros((2, 4))))
        self.assertTrue(torch.equal(item_before, torch.zeros((2, 8))))

        self.client.update(
            self.user_table, user_ids, torch.ones((2, 4), dtype=torch.float32)
        )
        user_after = self.client.pull(self.user_table, user_ids)
        item_after = self.client.pull(self.item_table, item_ids)
        self.assertTrue(torch.allclose(user_after, torch.full((2, 4), -0.01)))
        self.assertTrue(torch.equal(item_after, torch.zeros((2, 8))))

        encoded = self.client._encode_ids(self.item_table, item_ids)
        self.assertTrue(
            torch.equal(encoded >> _KEY_TAG_SHIFT, torch.tensor([1, 1]))
        )

    def test_row_overflow_is_rejected(self) -> None:
        self.client.init_data(
            name=self.user_table, shape=(4, 2), dtype=torch.float32
        )
        self.client.init_data(
            name=self.item_table, shape=(4, 2), dtype=torch.float32
        )
        foreign = torch.tensor([(2 << _KEY_TAG_SHIFT) | 1], dtype=torch.int64)
        with self.assertRaises(RuntimeError):
            self.client.pull(self.item_table, foreign)


if __name__ == "__main__":
    unittest.main()
