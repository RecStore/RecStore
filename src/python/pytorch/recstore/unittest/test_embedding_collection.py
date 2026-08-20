import unittest

import torch

from torchrec_kv import RecStoreEmbeddingCollection
from ..optimizer import SparseSGD


class _FakeJaggedTensor:
    def __init__(self, values, lengths):
        self._values = values
        self._lengths = lengths

    def values(self):
        return self._values

    def lengths(self):
        return self._lengths


class _FakeKeyedJaggedTensor:
    def __init__(self, features):
        self._features = dict(features)

    def keys(self):
        return list(self._features.keys())

    def __getitem__(self, key):
        return self._features[key]


class _FakeKVClient:
    def __init__(self, embedding_dim=4):
        self.embedding_dim = embedding_dim
        self.backend_lr = 0.01
        self.init_data_calls = []
        self.pull_calls = []
        self.update_async_calls = []
        self.wait_calls = []
        self.applied_handles = []
        self.fail_once_handles = set()
        self._rows = {}
        self._next_handle = 1

    def init_data(self, **kwargs):
        self.init_data_calls.append(dict(kwargs))

    def set_rows(self, keys, values):
        keys = keys.to(torch.int64).cpu()
        values = values.to(torch.float32).cpu()
        for row, key in enumerate(keys):
            self._rows[int(key.item())] = values[row].clone()

    def pull(self, name, ids):
        ids = ids.to(torch.int64).cpu()
        self.pull_calls.append((name, ids.clone()))
        rows = []
        for key in ids:
            rows.append(
                self._rows.get(
                    int(key.item()),
                    torch.zeros(self.embedding_dim, dtype=torch.float32),
                )
            )
        if not rows:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        return torch.stack(rows, dim=0)

    def update_async(self, name, ids, grads):
        handle = self._next_handle
        self._next_handle += 1
        self.update_async_calls.append(
            (name, ids.detach().clone(), grads.detach().clone(), handle)
        )
        return handle

    def wait(self, handle):
        if handle in self.fail_once_handles:
            self.fail_once_handles.remove(handle)
            raise RuntimeError("backend failure")
        self.wait_calls.append(int(handle))
        for _, ids, grads, queued_handle in self.update_async_calls:
            if queued_handle != handle:
                continue
            for key, grad in zip(ids.cpu(), grads.cpu()):
                row_id = int(key.item())
                current = self._rows.get(
                    row_id,
                    torch.zeros(self.embedding_dim, dtype=torch.float32),
                )
                self._rows[row_id] = current - self.backend_lr * grad
            self.applied_handles.append(handle)
            break


class _LegacyMetadataKVClient:
    def __init__(self):
        self.init_data_calls = []
        self.register_tensor_meta_calls = []
        self.init_embedding_table_calls = []

    def init_data(self, name, shape, dtype, base_offset=0):
        self.init_data_calls.append((name, shape, dtype, base_offset))

    def register_tensor_meta(self, **kwargs):
        self.register_tensor_meta_calls.append(dict(kwargs))

    def init_embedding_table(self, table_name, num_embeddings, embedding_dim):
        self.init_embedding_table_calls.append(
            (table_name, num_embeddings, embedding_dim)
        )
        return True


class TestRecStoreEmbeddingCollection(unittest.TestCase):
    def test_passes_custom_initializer_to_kv_client(self):
        fake = _FakeKVClient()

        def initializer(shape, dtype):
            return torch.ones(shape, dtype=dtype)

        RecStoreEmbeddingCollection(
            [
                {
                    "name": "t0",
                    "embedding_dim": 4,
                    "num_embeddings": 8,
                    "feature_names": ["f1"],
                }
            ],
            kv_client=fake,
            initialize_values=True,
            init_func=initializer,
        )

        self.assertTrue(fake.init_data_calls[0]["initialize_values"])
        self.assertIs(fake.init_data_calls[0]["init_func"], initializer)

    def test_forward_returns_jagged_tensors_with_fused_ids(self):
        fake = _FakeKVClient()
        configs = [
            {
                "name": "t0",
                "embedding_dim": 4,
                "num_embeddings": 8,
                "feature_names": ["f1"],
            },
            {
                "name": "t1",
                "embedding_dim": 4,
                "num_embeddings": 8,
                "feature_names": ["f2"],
            },
        ]
        ec = RecStoreEmbeddingCollection(
            configs,
            kv_client=fake,
            enable_fusion=True,
            fusion_k=4,
        )
        fake.set_rows(
            torch.tensor([1, 2, 19], dtype=torch.int64),
            torch.tensor(
                [
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                    [3.0, 3.1, 3.2, 3.3],
                ],
                dtype=torch.float32,
            ),
        )
        features = _FakeKeyedJaggedTensor(
            {
                "f1": _FakeJaggedTensor(
                    torch.tensor([1, 2], dtype=torch.int64),
                    torch.tensor([2], dtype=torch.int32),
                ),
                "f2": _FakeJaggedTensor(
                    torch.tensor([3], dtype=torch.int64),
                    torch.tensor([1], dtype=torch.int32),
                ),
                "payload": _FakeJaggedTensor(
                    torch.tensor([99], dtype=torch.int64),
                    torch.tensor([1], dtype=torch.int32),
                ),
            }
        )

        out = ec(features)

        self.assertEqual(set(out.keys()), {"f1", "f2"})
        self.assertTrue(
            torch.allclose(
                out["f1"].values(),
                torch.tensor(
                    [
                        [1.0, 1.1, 1.2, 1.3],
                        [2.0, 2.1, 2.2, 2.3],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                out["f2"].values(),
                torch.tensor([[3.0, 3.1, 3.2, 3.3]], dtype=torch.float32),
            )
        )
        self.assertTrue(torch.equal(out["f1"].lengths(), torch.tensor([2], dtype=torch.int32)))
        self.assertTrue(torch.equal(out["f2"].lengths(), torch.tensor([1], dtype=torch.int32)))
        self.assertEqual(len(fake.pull_calls), 2)
        self.assertEqual(fake.pull_calls[0][0], "t0")
        self.assertTrue(torch.equal(fake.pull_calls[0][1], torch.tensor([1, 2], dtype=torch.int64)))
        self.assertEqual(fake.pull_calls[1][0], "t0")
        self.assertTrue(torch.equal(fake.pull_calls[1][1], torch.tensor([19], dtype=torch.int64)))
        self.assertEqual(len(fake.init_data_calls), 2)
        self.assertFalse(fake.init_data_calls[0]["initialize_values"])
        self.assertFalse(fake.init_data_calls[1]["initialize_values"])
        self.assertEqual(fake.init_data_calls[0]["base_offset"], 0)
        self.assertEqual(fake.init_data_calls[1]["base_offset"], 16)

    def test_backward_trace_is_consumed_by_sparse_sgd(self):
        fake = _FakeKVClient()
        ec = RecStoreEmbeddingCollection(
            [
                {
                    "name": "t0",
                    "embedding_dim": 4,
                    "num_embeddings": 8,
                    "feature_names": ["f1"],
                }
            ],
            kv_client=fake,
            enable_fusion=False,
        )
        fake.set_rows(
            torch.tensor([1, 3], dtype=torch.int64),
            torch.tensor(
                [
                    [1.0, 1.1, 1.2, 1.3],
                    [3.0, 3.1, 3.2, 3.3],
                ],
                dtype=torch.float32,
            ),
        )
        features = _FakeKeyedJaggedTensor(
            {
                "f1": _FakeJaggedTensor(
                    torch.tensor([1, 1, 3], dtype=torch.int64),
                    torch.tensor([3], dtype=torch.int32),
                )
            }
        )

        out = ec(features)
        out["f1"].values().sum().backward()

        self.assertEqual(len(ec._trace), 1)
        self.assertEqual(ec._trace[0]["name"], "t0")
        self.assertTrue(
            torch.equal(ec._trace[0]["ids"], torch.tensor([1, 1, 3], dtype=torch.int64))
        )
        self.assertTrue(torch.allclose(ec._trace[0]["grads"], torch.ones((3, 4))))

        optimizer = SparseSGD([ec], lr=0.1)
        optimizer.step()
        self.assertTrue(
            torch.allclose(
                fake.pull("t0", torch.tensor([1, 3], dtype=torch.int64)),
                torch.tensor(
                    [
                        [1.0, 1.1, 1.2, 1.3],
                        [3.0, 3.1, 3.2, 3.3],
                    ]
                ),
            )
        )
        optimizer.flush()

        self.assertEqual(ec._trace, [])
        self.assertEqual(len(fake.update_async_calls), 1)
        table_name, ids, grads, handle = fake.update_async_calls[0]
        self.assertEqual(table_name, "t0")
        self.assertTrue(torch.equal(ids, torch.tensor([1, 3], dtype=torch.int64)))
        self.assertTrue(
            torch.allclose(
                grads,
                torch.tensor(
                    [
                        [2.0, 2.0, 2.0, 2.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(fake.wait_calls, [handle])
        self.assertTrue(
            torch.allclose(
                fake.pull("t0", torch.tensor([1, 3], dtype=torch.int64)),
                torch.tensor(
                    [
                        [0.98, 1.08, 1.18, 1.28],
                        [2.99, 3.09, 3.19, 3.29],
                    ]
                ),
            )
        )

    def test_flush_retries_only_unfinished_handles(self):
        fake = _FakeKVClient()
        ec = RecStoreEmbeddingCollection(
            [
                {
                    "name": "t0",
                    "embedding_dim": 4,
                    "num_embeddings": 8,
                    "feature_names": ["f1"],
                }
            ],
            kv_client=fake,
            enable_fusion=False,
        )
        optimizer = SparseSGD([ec], lr=0.1)
        first = fake.update_async(
            "t0", torch.tensor([1]), torch.ones((1, 4))
        )
        second = fake.update_async(
            "t0", torch.tensor([2]), torch.ones((1, 4))
        )
        optimizer._inflight_handles = [
            (fake, first, {"handle": first}),
            (fake, second, {"handle": second}),
        ]
        fake.fail_once_handles.add(second)

        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            optimizer.flush()

        self.assertEqual(fake.applied_handles, [first])
        self.assertEqual(
            [handle for _, handle, _ in optimizer._inflight_handles],
            [second],
        )

        optimizer.flush()
        self.assertEqual(fake.applied_handles, [first, second])
        self.assertEqual(optimizer._inflight_handles, [])

    def test_multiple_tables_require_fused_key_space(self):
        with self.assertRaisesRegex(ValueError, "require enable_fusion=True"):
            RecStoreEmbeddingCollection(
                [
                    {
                        "name": "t0",
                        "embedding_dim": 4,
                        "num_embeddings": 8,
                        "feature_names": ["f1"],
                    },
                    {
                        "name": "t1",
                        "embedding_dim": 4,
                        "num_embeddings": 8,
                        "feature_names": ["f2"],
                    },
                ],
                kv_client=_FakeKVClient(),
                enable_fusion=False,
            )

    def test_fused_table_ids_must_fit_the_prefix_width(self):
        with self.assertRaisesRegex(ValueError, "does not fit in fusion_k=4"):
            RecStoreEmbeddingCollection(
                [
                    {
                        "name": "t0",
                        "embedding_dim": 4,
                        "num_embeddings": 17,
                        "feature_names": ["f1"],
                    },
                    {
                        "name": "t1",
                        "embedding_dim": 4,
                        "num_embeddings": 8,
                        "feature_names": ["f2"],
                    },
                ],
                kv_client=_FakeKVClient(),
                enable_fusion=True,
                fusion_k=4,
            )

    def test_fusion_k_must_be_non_negative_for_one_table(self):
        with self.assertRaisesRegex(ValueError, "fusion_k must be non-negative"):
            RecStoreEmbeddingCollection(
                [
                    {
                        "name": "t0",
                        "embedding_dim": 4,
                        "num_embeddings": 8,
                        "feature_names": ["f1"],
                    }
                ],
                kv_client=_FakeKVClient(),
                enable_fusion=True,
                fusion_k=-1,
            )

    def test_initialize_without_values_uses_metadata_fallback_for_legacy_clients(self):
        fake = _LegacyMetadataKVClient()

        RecStoreEmbeddingCollection(
            [
                {
                    "name": "t0",
                    "embedding_dim": 4,
                    "num_embeddings": 8,
                    "feature_names": ["f1"],
                }
            ],
            kv_client=fake,
            enable_fusion=False,
            initialize_values=False,
        )

        self.assertEqual(fake.init_data_calls, [])
        self.assertEqual(len(fake.register_tensor_meta_calls), 1)
        self.assertEqual(fake.register_tensor_meta_calls[0]["name"], "t0")
        self.assertEqual(fake.register_tensor_meta_calls[0]["shape"], (8, 4))
        self.assertEqual(fake.init_embedding_table_calls, [("t0", 8, 4)])


if __name__ == "__main__":
    unittest.main()
