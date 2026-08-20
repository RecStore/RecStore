import copy

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from recstore.KVClient import get_kv_client
from torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection


class FakeOps:
    """Single-table fake backend for strict-step tests.

    The helper intentionally mirrors the current table-agnostic read path used by
    these tests; it should not be reused as a multi-table routing oracle.
    """

    def __init__(self):
        self._store = {}
        self.update_calls = []
        self.update_apply_count = 0
        self.last_applied_keys = None

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int, table_id: int = 0) -> int:
        return int(table_id)

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor):
        keys = keys.to(torch.int64).cpu().contiguous()
        values = values.to(torch.float32).cpu().contiguous()
        for i in range(keys.numel()):
            self._store[int(keys[i].item())] = values[i].clone()

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        keys = keys.to(torch.int64).cpu().contiguous()
        out = torch.zeros((keys.numel(), int(embedding_dim)), dtype=torch.float32)
        for i in range(keys.numel()):
            out[i] = self._store[int(keys[i].item())]
        return out

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor):
        keys = keys.to(torch.int64).cpu().contiguous()
        grads = grads.to(torch.float32).cpu().contiguous()
        self.update_calls.append((table_name, keys.clone(), grads.clone()))
        self.update_apply_count += 1
        self.last_applied_keys = keys.clone()
        for i in range(keys.numel()):
            self._store[int(keys[i].item())] -= grads[i]

    def read_rows(self, keys: torch.Tensor) -> torch.Tensor:
        return self.emb_read(keys, 4)


class KVClientIsolationMixin:
    def setUp(self):
        super().setUp()
        self.kv_client = get_kv_client()
        self._saved_state = {
            "tensor_meta": copy.deepcopy(self.kv_client._tensor_meta),
            "full_data_shape": copy.deepcopy(self.kv_client._full_data_shape),
            "data_name_list": set(self.kv_client._data_name_list),
            "gdata_name_list": set(self.kv_client._gdata_name_list),
            "pending_async_ops": copy.deepcopy(self.kv_client._pending_async_ops),
            "next_async_handle": self.kv_client._next_async_handle,
            "ops": self.kv_client.ops,
        }

    def tearDown(self):
        self.kv_client._tensor_meta = self._saved_state["tensor_meta"]
        self.kv_client._full_data_shape = self._saved_state["full_data_shape"]
        self.kv_client._data_name_list = self._saved_state["data_name_list"]
        self.kv_client._gdata_name_list = self._saved_state["gdata_name_list"]
        self.kv_client._pending_async_ops = self._saved_state["pending_async_ops"]
        self.kv_client._next_async_handle = self._saved_state["next_async_handle"]
        self.kv_client.ops = self._saved_state["ops"]
        super().tearDown()

    def build_module(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=8, feature_names=["f1"]),
        ]
        kv_client = self.kv_client
        kv_client._tensor_meta["t0"] = {"shape": (8, 4), "dtype": torch.float32}
        kv_client._full_data_shape["t0"] = (8, 4)
        kv_client._data_name_list.add("t0")
        kv_client._gdata_name_list.add("t0")
        kv_client._pending_async_ops = {}
        kv_client._next_async_handle = 1
        ebc = RecStoreEmbeddingBagCollection(configs, enable_fusion=False)
        fake = FakeOps()
        ebc.kv_client.ops = fake
        keys = torch.arange(8, dtype=torch.int64)
        values = torch.arange(32, dtype=torch.float32).view(8, 4)
        fake.emb_write(keys, values)
        return ebc, fake


def build_features():
    return KeyedJaggedTensor.from_lengths_sync(
        keys=["f1"],
        values=torch.tensor([1, 1, 3], dtype=torch.int64),
        lengths=torch.tensor([3], dtype=torch.int32),
    )
