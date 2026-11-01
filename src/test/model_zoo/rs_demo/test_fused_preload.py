import unittest
import warnings

import torch

from python.pytorch.torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection
from python.pytorch.torchrec_kv.EmbeddingCollection import RecStoreEmbeddingCollection


class _RecordingBase:
    """Stub with the shared client surface: meta registration and init_data.

    The signature deliberately omits ``initialize_backend``; the fused pre-load
    must work with the smaller interface both RecStore clients share.
    """

    def __init__(self) -> None:
        self.registered: list[tuple[str, int, int]] = []
        self.init_data_calls: list[tuple[str, int, int]] = []
        self.init_table_calls: list[tuple[str, int, int]] = []

    def register_tensor_meta(self, name, shape, dtype, base_offset=0):
        self.registered.append((name, int(shape[0]), int(base_offset)))
        return None

    def init_embedding_table(self, name, num_embeddings, embedding_dim):
        self.init_table_calls.append((name, int(num_embeddings), int(embedding_dim)))
        return 0

    def init_data(
        self,
        name,
        shape,
        dtype,
        base_offset=0,
        init_func=None,
        initialize_values=True,
    ):
        self.init_data_calls.append((name, int(shape[0]), int(base_offset)))
        return None


class _RecordingClient(_RecordingBase):
    """Stub that can write rows into the master key space."""

    def __init__(self) -> None:
        super().__init__()
        self.pushes: list[tuple[str, int, int, int]] = []
        self.pushed_values: list[torch.Tensor] = []

    def push(self, name, ids, data):
        keys = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids)
        self.pushes.append((name, int(keys.numel()), int(keys.min()), int(keys.max())))
        self.pushed_values.append(data)
        return None


class _NoPushClient(_RecordingBase):
    """Stub without ``push``: rows can only be registered, not pre-created."""


def _bag_collection(enable_fusion: bool, client) -> RecStoreEmbeddingBagCollection:
    return RecStoreEmbeddingBagCollection(
        [
            {
                "name": f"t_f{i}",
                "num_embeddings": size,
                "embedding_dim": 4,
                "feature_names": [f"f{i}"],
            }
            for i, size in enumerate((1000, 40, 7))
        ],
        enable_fusion=enable_fusion,
        fusion_k=30,
        kv_client=client,
    )


def _collection(
    enable_fusion: bool,
    client,
    *,
    initialize_values: bool = False,
    init_func=None,
) -> RecStoreEmbeddingCollection:
    return RecStoreEmbeddingCollection(
        [
            {
                "name": f"c_f{i}",
                "num_embeddings": size,
                "embedding_dim": 4,
                "feature_names": [f"g{i}"],
            }
            for i, size in enumerate((1000, 40, 7))
        ],
        enable_fusion=enable_fusion,
        fusion_k=30,
        kv_client=client,
        initialize_values=initialize_values,
        init_func=init_func,
    )


class FusedBagPreloadTest(unittest.TestCase):
    def test_fused_preload_writes_every_feature_into_the_master_key_space(self) -> None:
        client = _RecordingClient()
        _bag_collection(enable_fusion=True, client=client)

        # The first table is pre-created through the ordinary path, the rest are
        # pushed into the master table at their fused offsets.
        self.assertEqual([("t_f0", 1000, 0)], client.init_data_calls)
        self.assertEqual(
            [("t_f1", 40, 1 << 30), ("t_f2", 7, 2 << 30)],
            client.registered,
        )
        # The feature tables still exist on the backend so that every rank
        # opens the same set of PS tables during setup.
        self.assertEqual(
            [("t_f1", 40, 4), ("t_f2", 7, 4)],
            client.init_table_calls,
        )
        self.assertEqual(
            [
                ("t_f0", 40, 1 << 30, (1 << 30) + 39),
                ("t_f0", 7, 2 << 30, (2 << 30) + 6),
            ],
            client.pushes,
        )

    def test_without_fusion_each_table_is_initialised_in_place(self) -> None:
        client = _RecordingClient()
        _bag_collection(enable_fusion=False, client=client)

        self.assertEqual(
            [("t_f0", 1000, 0), ("t_f1", 40, 0), ("t_f2", 7, 0)],
            client.init_data_calls,
        )
        self.assertEqual([], client.pushes)

    def test_client_without_push_registers_rows_instead_of_failing(self) -> None:
        client = _NoPushClient()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _bag_collection(enable_fusion=True, client=client)

        self.assertEqual(
            [("t_f1", 40, 1 << 30), ("t_f2", 7, 2 << 30)],
            client.registered,
        )
        self.assertFalse(hasattr(client, "push"))
        self.assertTrue(
            any("not pre-created" in str(w.message) for w in caught),
            "a client without push() should report that rows stay lazy",
        )


class FusedCollectionPreloadTest(unittest.TestCase):
    def test_default_initialization_registers_metadata_only(self) -> None:
        client = _RecordingClient()
        _collection(enable_fusion=True, client=client)

        # initialize_values defaults to False: register the offsets, write
        # nothing, and leave row creation to the first update.
        self.assertEqual([("c_f0", 1000, 0)], client.init_data_calls)
        self.assertEqual(
            [("c_f1", 40, 1 << 30), ("c_f2", 7, 2 << 30)],
            client.registered,
        )
        self.assertEqual([], client.pushes)

    def test_custom_initialization_writes_into_the_master_key_space(self) -> None:
        client = _RecordingClient()

        def _sevens(shape, dtype):
            return torch.full(shape, 7.0, dtype=dtype)

        _collection(
            enable_fusion=True,
            client=client,
            initialize_values=True,
            init_func=_sevens,
        )

        self.assertEqual(
            [
                ("c_f0", 40, 1 << 30, (1 << 30) + 39),
                ("c_f0", 7, 2 << 30, (2 << 30) + 6),
            ],
            client.pushes,
        )
        for values in client.pushed_values:
            self.assertTrue(torch.all(values == 7.0))

    def test_without_fusion_the_table_is_initialised_in_place(self) -> None:
        client = _RecordingClient()
        # Non-fused collections support a single table only.
        RecStoreEmbeddingCollection(
            [
                {
                    "name": "c_f0",
                    "num_embeddings": 1000,
                    "embedding_dim": 4,
                    "feature_names": ["g0"],
                }
            ],
            enable_fusion=False,
            fusion_k=30,
            kv_client=client,
        )

        self.assertEqual([("c_f0", 1000, 0)], client.init_data_calls)
        self.assertEqual([], client.pushes)


if __name__ == "__main__":
    unittest.main()
