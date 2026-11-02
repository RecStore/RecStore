"""CPU checks for the explicit, fresh-table EBC namespace layout."""
import copy
import unittest

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..KVClient import RecStoreClient
from ._sparse_test_utils import FakeOps
from torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection


class NamespaceOps(FakeOps):
    """Match backend name/config checks and optimizer-specific parameter tags."""

    def __init__(self, roles=1):
        super().__init__()
        self.roles = roles
        self.tables = {}
        self.init_calls = []
        self.write_calls = 0

    def init_embedding_table(self, name, rows, dim, table_id=0):
        config = rows, dim, table_id
        if name in self.tables and self.tables[name] != config:
            raise RuntimeError("backend table configuration mismatch")
        self.tables[name] = config
        self.init_calls.append((name, config))
        return table_id * self.roles

    def emb_write(self, keys, values):
        self.write_calls += 1
        super().emb_write(keys, values)

    def load_checkpoint(self, path, metadata):
        return True


def client(ops=None):
    kv = object.__new__(RecStoreClient)
    kv._initialized = False
    RecStoreClient.__init__(kv)
    kv.ops = ops if ops is not None else NamespaceOps()
    return kv


def configs(prefix):
    return [dict(name=f"{prefix}_t{i}", num_embeddings=4, embedding_dim=2,
                 feature_names=[f"{prefix}_f{i}"]) for i in range(2)]


def build(kv, prefix="a", **kwargs):
    return RecStoreEmbeddingBagCollection(
        configs(prefix), kv_client=kv, fusion_k=4, fast_path_mode="off",
        shared_fused_namespace=True, **kwargs)


def set_initial_values(kv, values):
    original = kv.init_data

    def initialize(**kwargs):
        value = values[kwargs["name"]]
        return original(**kwargs, init_func=lambda shape, dtype: torch.full(shape, value, dtype=dtype))

    kv.init_data = initialize


def pooled(module, prefix):
    features = KeyedJaggedTensor.from_lengths_sync(
        keys=[f"{prefix}_f0", f"{prefix}_f1"], values=torch.tensor([1, 1]),
        lengths=torch.tensor([1, 1], dtype=torch.int32))
    return module(features).values().detach()


class TestSharedFusedNamespace(unittest.TestCase):
    def test_nonzero_initializer_and_multiple_collections_after_legacy_table(self):
        kv = client()
        kv.init_data("prior", (4, 2), torch.float32,
                     init_func=lambda shape, dtype: torch.full(shape, 3., dtype=dtype))
        set_initial_values(kv, {"a_t0": 7., "a_t1": 11., "b_t0": 13., "b_t1": 17.})
        first, second = build(kv, "a"), build(kv, "b")
        self.assertTrue(torch.equal(pooled(first, "a"), torch.tensor([[7., 7., 11., 11.]])))
        self.assertTrue(torch.equal(pooled(second, "b"), torch.tensor([[13., 13., 17., 17.]])))
        self.assertTrue(torch.equal(kv.pull("prior", torch.tensor([1])), torch.tensor([[3., 3.]])))
        self.assertEqual([kv.get_embedding_table_namespace(n)[0]
                          for n in ["prior", "a_t0", "a_t1", "b_t0", "b_t1"]], [0, 1, 1, 2, 2])
        self.assertEqual(kv._next_table_id, 3)
        self.assertEqual(len(kv.ops._store), 20)
        self.assertEqual(first.shared_fused_namespace_identity()["table_id"], 1)

    def test_metadata_rank_registers_without_writes_and_matches_explicit_master(self):
        ops = NamespaceOps(roles=3)
        owner = client(ops)
        set_initial_values(owner, {"a_t0": 7., "a_t1": 11.})
        source = build(owner, backend_table_id=5)
        writes = ops.write_calls
        reader = client(ops)
        replica = build(reader, initialize_tables=False, backend_table_id=5)
        self.assertEqual(ops.write_calls, writes)
        self.assertEqual(len(ops.init_calls), 4)
        self.assertEqual(source.shared_fused_namespace_identity(), replica.shared_fused_namespace_identity())
        self.assertEqual(reader.get_embedding_table_namespace("a_t1"), (5, 15))
        self.assertTrue(torch.equal(pooled(replica, "a"), torch.tensor([[7., 7., 11., 11.]])))

    def test_optimizer_role_tags_are_not_table_ids_and_state_rows_are_separate(self):
        for roles in (1, 2, 3):
            with self.subTest(roles=roles):
                kv = client(NamespaceOps(roles))
                set_initial_values(kv, {"a_t0": 7., "a_t1": 11.})
                module = build(kv, backend_table_id=5)
                tag = 5 * roles
                self.assertEqual(kv.get_embedding_table_namespace("a_t0"), (5, tag))
                self.assertEqual(kv.get_embedding_table_namespace("a_t1"), (5, tag))
                for role in range(1, roles):
                    kv.ops._store[((tag + role) << 56) | 17] = torch.full((2,), 100. + role)
                kv.update("a_t0", torch.tensor([17]), torch.full((1, 2), 3.))
                self.assertTrue(torch.equal(pooled(module, "a"), torch.tensor([[7., 7., 8., 8.]])))
                for role in range(1, roles):
                    self.assertTrue(torch.equal(kv.ops._store[((tag + role) << 56) | 17], torch.full((2,), 100. + role)))

    def test_explicit_metadata_registration_is_idempotent_and_rejects_conflicts(self):
        kv = client(NamespaceOps(3))
        kwargs = dict(name="t", shape=(4, 2), dtype=torch.float32,
                      base_offset=16, backend_table_id=5)
        kv.init_data(**kwargs)
        before, writes = copy.deepcopy(kv._tensor_meta), kv.ops.write_calls
        kv.init_data(**kwargs)
        self.assertEqual(kv.ops.write_calls, writes)
        self.assertEqual(kv.get_embedding_table_namespace("t"), (5, 15))
        for override in (dict(shape=(5, 2)), dict(dtype=torch.float64),
                         dict(base_offset=32), dict(backend_table_id=6),
                         dict(backend_table_id=True), dict(backend_table_id=-1)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                kv.init_data(**(kwargs | override))
        with self.assertRaises(ValueError):
            kv.init_embedding_table("t", 4, 2, table_id=6)
        self.assertEqual(kv._tensor_meta, before)
        self.assertEqual(kv.ops.write_calls, writes)
        with self.assertRaises(ValueError):
            kv.register_tensor_meta("unregistered", (4, 2), torch.float32, backend_table_id=7)

    def test_metadata_only_explicit_registration_and_tag_drift(self):
        kv = client(NamespaceOps(3))
        kv.register_tensor_meta("t", (4, 2), torch.float32, base_offset=16,
                                initialize_backend=True, backend_table_id=5)
        self.assertEqual(kv.get_embedding_table_namespace("t"), (5, 15))
        self.assertEqual(kv.ops.write_calls, 0)
        before = copy.deepcopy(kv._tensor_meta)
        kv.ops.roles = 2
        with self.assertRaisesRegex(ValueError, "parameter tag"):
            kv.init_embedding_table("t", 4, 2, table_id=5)
        self.assertEqual(kv._tensor_meta, before)
        kv.register_tensor_meta("local_only", (4, 2), torch.float32)
        with self.assertRaises(RuntimeError):
            kv.get_embedding_table_namespace("local_only")
        with self.assertRaises(ValueError):
            kv.register_tensor_meta("local_only", (4, 2), torch.float32,
                                    initialize_backend=True, backend_table_id=6)

    def test_shared_layout_rejects_existing_names_and_namespace_collisions(self):
        kv = client()
        kv.init_data("a_t0", (4, 2), torch.float32)
        before = copy.deepcopy(kv._tensor_meta)
        with self.assertRaisesRegex(ValueError, "unregistered table names"):
            build(kv)
        with self.assertRaisesRegex(ValueError, "already used"):
            build(kv, "b", backend_table_id=0)
        self.assertEqual(kv._tensor_meta, before)
        self.assertEqual(set(kv.ops.tables), {"a_t0"})
        unknown = client()
        unknown.register_tensor_meta("legacy_metadata", (4, 2), torch.float32)
        with self.assertRaisesRegex(RuntimeError, "no registered backend namespace"):
            build(unknown)
        self.assertFalse(unknown.ops.tables)

    def test_invalid_fusion_layout_is_rejected_before_backend_registration(self):
        cases = [dict(enable_fusion=False), dict(fusion_k=1), dict(fusion_k=56)]
        for override in cases:
            with self.subTest(override=override):
                kv = client()
                with self.assertRaises(ValueError):
                    RecStoreEmbeddingBagCollection(configs("a"), kv_client=kv,
                        shared_fused_namespace=True, **({"fusion_k": 4} | override))
                self.assertFalse(kv.ops.tables)
        invalid = configs("a")
        invalid[1]["embedding_dim"] = 3
        with self.assertRaises(ValueError):
            RecStoreEmbeddingBagCollection(invalid, kv_client=client(), shared_fused_namespace=True)

    def test_checkpoint_layout_is_rejected_in_both_creation_orders(self):
        metadata = {"identity": {}, "checkpoint_id": "legacy"}
        kv = client()
        module = build(kv)
        self.assertEqual(module.shared_fused_namespace_identity()["layout"], "shared_fused_namespace_v1")
        for method in (kv.save_checkpoint, kv.load_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "unsupported.*shared_fused_namespace_v1"):
                method("unused", metadata)
        loaded = client()
        loaded.load_checkpoint("unused", metadata)
        with self.assertRaisesRegex(RuntimeError, "legacy checkpoint"):
            build(loaded)
        self.assertFalse(loaded.ops.tables)

    def test_legacy_default_still_allocates_separate_table_namespaces(self):
        kv = client()
        RecStoreEmbeddingBagCollection(configs("a"), kv_client=kv, fusion_k=4)
        self.assertEqual(kv.get_embedding_table_namespace("a_t0"), (0, 0))
        self.assertEqual(kv.get_embedding_table_namespace("a_t1"), (1, 1))
        self.assertFalse(getattr(kv, "_shared_fused_namespace", False))


if __name__ == "__main__":
    unittest.main()
