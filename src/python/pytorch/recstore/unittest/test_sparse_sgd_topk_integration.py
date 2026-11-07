import pytest
import torch

import recstore.optimizer as optimizer_module
from recstore.optimizer import SparseSGD


class FakeKVClient:
    """Record update requests without sending RPCs."""

    def __init__(self):
        self.calls = []
        self.waited = []

    def current_ps_backend(self):
        return "grpc"

    def update_async(self, name, ids, grads, learning_rate=0.0):
        self.calls.append((
            name,
            ids.detach().clone(),
            grads.detach().clone(),
        ))
        return len(self.calls)

    def wait(self, handle):
        self.waited.append(handle)


class FakeEmbedding:
    """Represent an embedding module with fusion enabled."""

    def __init__(self, client):
        self.kv_client = client
        self._enable_fusion = True
        self.fast_path_mode = "off"
        self._config_names = {"feature": "master"}

        self._trace = [
            (
                "master",
                torch.tensor([10, 20]),
                torch.tensor([
                    [1.0, 2.0],
                    [0.1, 0.0],
                ]),
            ),
            (
                "master",
                torch.tensor([10, 30]),
                torch.tensor([
                    [5.0, 6.0],
                    [0.0, 2.0],
                ]),
            ),
        ]

    def resolve_fast_path_backend(self):
        return None

    def reset_trace(self):
        self._trace.clear()


def test_sparse_sgd_sends_topk_gradients():
    client = FakeKVClient()
    embedding = FakeEmbedding(client)

    optimizer = SparseSGD(
        params=[embedding],
        lr=0.01,
        keep_ratio=0.5,
    )

    optimizer.step()

    # Submit one update to the KV client.
    assert len(client.calls) == 1

    name, ids, grads = client.calls[0]

    assert name == "master"

    # Sum ID 10 to [6, 8], then keep the two strongest of three unique rows.
    torch.testing.assert_close(
        ids,
        torch.tensor([10, 30]),
    )

    torch.testing.assert_close(
        grads,
        torch.tensor([
            [6.0, 8.0],
            [0.0, 2.0],
        ]),
    )

    # Wait for the submitted update.
    optimizer.flush()
    assert client.waited == [1]


def test_sparse_sgd_default_no_compression():
    client = FakeKVClient()
    embedding = FakeEmbedding(client)

    # The default keep_ratio is 1.0.
    optimizer = SparseSGD(
        params=[embedding],
        lr=0.01,
    )

    optimizer.step()

    assert len(client.calls) == 1

    name, ids, grads = client.calls[0]

    assert name == "master"

    # Aggregate duplicate IDs without dropping any gradient rows.
    torch.testing.assert_close(
        ids,
        torch.tensor([10, 20, 30]),
    )

    torch.testing.assert_close(
        grads,
        torch.tensor([
            [6.0, 8.0],
            [0.1, 0.0],
            [0.0, 2.0],
        ]),
    )

    optimizer.flush()
    assert client.waited == [1]

class FakeLocalKVClient(FakeKVClient):
    """Record updates for a Local SHM backend."""

    def __init__(self):
        super().__init__()
        self.local_calls = []

    def current_ps_backend(self):
        return "local_shm"

    def is_shared_local_shm_table(self):
        return True

    def local_update_flat(self, name, ids, grads, learning_rate=0.0):
        self.local_calls.append((
            name,
            ids.detach().clone(),
            grads.detach().clone(),
        ))


class FakeLocalEmbedding(FakeEmbedding):
    """Represent a module eligible for the Local SHM fast path."""

    def __init__(self, client):
        super().__init__(client)
        self.fast_path_mode = "auto"
        self._enable_fusion = True
        self._master_config = object()


def test_local_shm_topk_compression():
    client = FakeLocalKVClient()
    embedding = FakeLocalEmbedding(client)

    optimizer = SparseSGD(
        params=[embedding],
        lr=0.01,
        keep_ratio=0.5,
    )

    optimizer.step()

    # Use the Local SHM update path.
    assert len(client.local_calls) == 1

    # The asynchronous RPC path must remain unused.
    assert len(client.calls) == 0

    name, ids, grads = client.local_calls[0]

    assert name == "master"

    # Keep two of the three unique rows after aggregation.
    torch.testing.assert_close(
        ids,
        torch.tensor([10, 30]),
    )

    torch.testing.assert_close(
        grads,
        torch.tensor([
            [6.0, 8.0],
            [0.0, 2.0],
        ]),
    )

    # Local SHM records the completed update payload immediately.
    payloads = optimizer.last_update_payloads()
    assert len(payloads) == 1
    torch.testing.assert_close(payloads[0]["ids"], ids)
    torch.testing.assert_close(payloads[0]["grads"], grads)


def test_sparse_sgd_distributed_grpc_topk():
    client = FakeKVClient()

    # Use the distributed gRPC backend name reported in DLRM runs.
    client.current_ps_backend = lambda: "distributed_grpc"

    embedding = FakeEmbedding(client)

    optimizer = SparseSGD(
        params=[embedding],
        lr=0.01,
        keep_ratio=0.5,
    )

    optimizer.step()

    assert len(client.calls) == 1

    name, ids, grads = client.calls[0]

    assert name == "master"
    torch.testing.assert_close(ids, torch.tensor([10, 30]))
    torch.testing.assert_close(
        grads,
        torch.tensor([[6.0, 8.0], [0.0, 2.0]]),
    )

    optimizer.flush()
    assert client.waited == [1]



def test_dist_embedding_rejects_topk(monkeypatch):
    client = FakeKVClient()

    class FakeDistEmbedding:
        def __init__(self):
            self.kv_client = client

    # Make the optimizer recognize the test module as DistEmbedding.
    monkeypatch.setattr(
        optimizer_module,
        "DistEmbedding",
        FakeDistEmbedding,
    )

    optimizer = optimizer_module.SparseSGD(
        params=[FakeDistEmbedding()],
        lr=0.01,
        keep_ratio=0.25,
    )

    with pytest.raises(NotImplementedError, match="DistEmbedding"):
        optimizer.step()

    # No update should reach the KV client.
    assert client.calls == []


def test_topk_rejects_unsupported_module_before_updates(monkeypatch):
    client = FakeKVClient()

    supported_module = FakeEmbedding(client)

    class FakeDistEmbedding:
        def __init__(self):
            self.kv_client = client

    monkeypatch.setattr(
        optimizer_module,
        "DistEmbedding",
        FakeDistEmbedding,
    )

    unsupported_module = FakeDistEmbedding()

    optimizer = optimizer_module.SparseSGD(
        params=[supported_module, unsupported_module],
        lr=0.01,
        keep_ratio=0.5,
    )

    with pytest.raises(NotImplementedError, match="DistEmbedding"):
        optimizer.step()

    # Reject the second module before submitting the first module's update.
    assert client.calls == []
