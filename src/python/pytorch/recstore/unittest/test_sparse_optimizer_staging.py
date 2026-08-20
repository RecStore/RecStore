import unittest
from unittest import mock

import torch

from ..optimizer import SparseSGD
from ._sparse_test_utils import KVClientIsolationMixin, build_features


class TestSparseOptimizerStaging(KVClientIsolationMixin, unittest.TestCase):
    def test_failed_wait_keeps_update_available_for_retry(self):
        ebc, fake = self.build_module()
        client = ebc.kv_client
        handle = client.update_async(
            "t0",
            torch.tensor([1], dtype=torch.int64),
            torch.ones((1, 4), dtype=torch.float32),
        )
        apply_update = fake.emb_update_table

        def fail_update(*_args, **_kwargs):
            raise RuntimeError("backend failure")

        fake.emb_update_table = fail_update
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            client.wait(handle)
        self.assertIn(handle, client._pending_async_ops)

        fake.emb_update_table = apply_update
        client.wait(handle)
        self.assertNotIn(handle, client._pending_async_ops)
        with self.assertRaisesRegex(RuntimeError, "Unknown async update handle"):
            client.wait(handle)

    def test_rdma_update_is_submitted_before_wait(self):
        ebc, fake = self.build_module()
        submitted = {}
        wait_calls = []

        fake.current_ps_backend = lambda: "rdma"

        def submit(name, ids, grads):
            submitted[77] = (name, ids.clone(), grads.clone())
            return 77

        def wait(handle):
            wait_calls.append(int(handle))
            name, ids, grads = submitted.pop(int(handle))
            fake.emb_update_table(name, ids, grads)

        fake.emb_update_async = submit
        fake.emb_update_wait = wait
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        optimizer.zero_grad()
        ebc(features).values().sum().backward()
        with mock.patch.object(
            torch,
            "unique",
            side_effect=AssertionError("RDMA SGD applies duplicate rows directly"),
        ):
            optimizer.step()

        self.assertIn(77, submitted)
        self.assertEqual(fake.update_calls, [])
        self.assertTrue(torch.equal(submitted[77][1], torch.tensor([1, 1, 3])))

        optimizer.flush()

        self.assertEqual(wait_calls, [77])
        self.assertEqual(len(fake.update_calls), 1)

    def test_sparse_optimizer_uses_module_specific_kv_client_for_async_updates(self):
        class _ProxyKVClient:
            def __init__(self, inner):
                self._inner = inner
                self.update_async_calls = []
                self.wait_calls = []
                self._next_handle = 100

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def update_async(self, name, ids, grads):
                handle = self._next_handle
                self._next_handle += 1
                self.update_async_calls.append((name, ids.clone(), grads.clone(), handle))
                return handle

            def wait(self, handle):
                self.wait_calls.append(int(handle))

        ebc, fake = self.build_module()
        proxy_client = _ProxyKVClient(ebc.kv_client)
        ebc.kv_client = proxy_client
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()

        optimizer.step()
        optimizer.flush()

        self.assertEqual(len(proxy_client.update_async_calls), 1)
        self.assertEqual(proxy_client.wait_calls, [100])
        self.assertEqual(fake.update_calls, [])

    def test_ebc_update_submits_unscaled_grads_for_backend_optimizer(self):
        ebc, fake = self.build_module()
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()

        self.assertEqual(fake.update_calls, [])

        optimizer.step()
        optimizer.flush()

        self.assertEqual(len(fake.update_calls), 1)
        table_name, keys, grads = fake.update_calls[0]
        self.assertEqual(table_name, "t0")
        self.assertTrue(torch.equal(keys, torch.tensor([1, 3], dtype=torch.int64)))
        expected_grads = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(grads, expected_grads))

    def test_sparse_update_not_visible_before_flush(self):
        ebc, fake = self.build_module()
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        baseline = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))

        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()

        optimizer.step()

        self.assertEqual(len(fake.update_calls), 0)
        self.assertEqual(fake.update_apply_count, 0)
        current = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))
        self.assertTrue(torch.allclose(current, baseline))

    def test_sparse_update_becomes_visible_after_flush(self):
        ebc, fake = self.build_module()
        optimizer = SparseSGD([ebc], lr=0.1)
        features = build_features()

        baseline = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))

        optimizer.zero_grad()
        result = ebc(features)
        loss = result.values().sum()
        loss.backward()

        optimizer.step()
        optimizer.flush()

        self.assertEqual(fake.update_apply_count, 1)
        self.assertEqual(len(fake.update_calls), 1)
        self.assertTrue(torch.equal(fake.last_applied_keys, torch.tensor([1, 3], dtype=torch.int64)))
        current = fake.read_rows(torch.tensor([1, 3], dtype=torch.int64))
        self.assertFalse(torch.allclose(current, baseline))


if __name__ == "__main__":
    unittest.main()
