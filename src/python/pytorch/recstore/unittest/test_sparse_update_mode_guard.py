import unittest

import torch

from ..optimizer import (
    SparseSGD,
    _CLIENT_PUSH_MODE,
    _SERVER_UPDATE_MODE,
    _process_dist_embedding_module,
    _record_table_update_mode,
)


class _FakeKvClient:
    def __init__(self) -> None:
        self.update_calls = []

    def update_async(self, name, ids, grads, learning_rate=0.0):
        self.update_calls.append((name, ids.clone(), grads.clone()))
        return len(self.update_calls)

    def _perf_add(self, *args) -> None:
        pass


class _FakeTracedModule:
    """Minimal module accepted by the generic SparseSGD trace path."""

    _config_names = ("t0",)
    _enable_fusion = False
    fast_path_mode = "off"

    def __init__(self, client) -> None:
        self.kv_client = client
        self._trace = [
            {
                "name": "t0",
                "ids": torch.tensor([1, 1, 2], dtype=torch.int64),
                "grads": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            }
        ]

    def resolve_fast_path_backend(self):
        return None

    def reset_trace(self) -> None:
        self._trace = []


class _FakeWeight:
    def __getitem__(self, ids):
        return torch.zeros((ids.numel(), 2))

    def __setitem__(self, ids, values) -> None:
        pass


class _FakeDistEmbedding:
    name = "t0"
    embedding_dim = 2

    def __init__(self) -> None:
        self._trace = [(torch.tensor([1, 2]), torch.ones((2, 2)))]
        self.weight = _FakeWeight()


class TestTableUpdateModeGuard(unittest.TestCase):
    def test_same_mode_is_allowed_repeatedly(self) -> None:
        client = _FakeKvClient()

        _record_table_update_mode(client, "t0", _CLIENT_PUSH_MODE)
        _record_table_update_mode(client, "t0", _CLIENT_PUSH_MODE)

    def test_different_modes_on_one_table_raise(self) -> None:
        client = _FakeKvClient()

        _record_table_update_mode(client, "t0", _CLIENT_PUSH_MODE)

        with self.assertRaisesRegex(RuntimeError, "mixing"):
            _record_table_update_mode(client, "t0", _SERVER_UPDATE_MODE)

    def test_different_tables_are_independent(self) -> None:
        client = _FakeKvClient()

        _record_table_update_mode(client, "t0", _CLIENT_PUSH_MODE)
        _record_table_update_mode(client, "t1", _SERVER_UPDATE_MODE)

    def test_server_update_then_client_push_raises(self) -> None:
        client = _FakeKvClient()
        optimizer = SparseSGD([_FakeTracedModule(client)], lr=0.1)
        optimizer.step()
        self.assertEqual(len(client.update_calls), 1)

        with self.assertRaisesRegex(RuntimeError, "mixing"):
            _process_dist_embedding_module(
                _FakeDistEmbedding(), 0.1, client
            )

    def test_client_push_then_server_update_raises(self) -> None:
        client = _FakeKvClient()
        _process_dist_embedding_module(_FakeDistEmbedding(), 0.1, client)

        optimizer = SparseSGD([_FakeTracedModule(client)], lr=0.1)
        with self.assertRaisesRegex(RuntimeError, "mixing"):
            optimizer.step()
        self.assertEqual(client.update_calls, [])

    def test_dist_module_without_trace_does_not_claim_the_table(self) -> None:
        client = _FakeKvClient()
        idle = _FakeDistEmbedding()
        idle._trace = []

        _process_dist_embedding_module(idle, 0.1, client)

        optimizer = SparseSGD([_FakeTracedModule(client)], lr=0.1)
        optimizer.step()
        self.assertEqual(len(client.update_calls), 1)


if __name__ == "__main__":
    unittest.main()
