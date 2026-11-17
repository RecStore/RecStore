"""Regression tests for all-or-reject Top-K sparse update configuration."""

from types import SimpleNamespace

import pytest
import torch

import recstore.optimizer as optimizer_module
from recstore.optimizer import SparseSGD


class RecordingKVClient:
    def __init__(self, backend="distributed_grpc"):
        self.backend = backend
        self.calls = []

    def current_ps_backend(self):
        return self.backend

    def update_async(self, name, ids, grads, learning_rate=0.0):
        self.calls.append((name, ids.detach().clone(), grads.detach().clone()))
        return len(self.calls)

    def wait(self, handle):
        pass


class RecordingEmbedding:
    def __init__(self, client, *, fusion=True, fast_path=None):
        self.kv_client = client
        self._config_names = {"feature": "master"}
        self._enable_fusion = fusion
        self.fast_path_mode = "off"
        self._fast_path = fast_path
        self._trace = [
            (
                "master",
                torch.tensor([10, 20, 10, 30]),
                torch.tensor(
                    [[1.0, 2.0], [0.1, 0.0], [5.0, 6.0], [0.0, 2.0]]
                ),
            )
        ]

    def resolve_fast_path_backend(self):
        return self._fast_path

    def reset_trace(self):
        self._trace.clear()


def test_strict_topk_updates_supported_distributed_grpc():
    client = RecordingKVClient()
    optimizer = SparseSGD(
        [RecordingEmbedding(client)], lr=0.01, keep_ratio=0.5
    )

    optimizer.step()

    assert len(client.calls) == 1
    name, ids, grads = client.calls[0]
    assert name == "master"
    torch.testing.assert_close(ids, torch.tensor([10, 30]))
    torch.testing.assert_close(grads, torch.tensor([[6.0, 8.0], [0.0, 2.0]]))


def test_strict_topk_validates_all_modules_before_submitting(monkeypatch):
    client = RecordingKVClient()
    supported = RecordingEmbedding(client)

    class UnsupportedDistEmbedding:
        def __init__(self):
            self.kv_client = client

    monkeypatch.setattr(
        optimizer_module, "DistEmbedding", UnsupportedDistEmbedding
    )
    optimizer = SparseSGD(
        [supported, UnsupportedDistEmbedding()], lr=0.01, keep_ratio=0.5
    )

    with pytest.raises(NotImplementedError, match="DistEmbedding"):
        optimizer.step()
    assert client.calls == []


def test_strict_topk_rejects_non_fused_module_before_any_updates():
    client = RecordingKVClient()
    optimizer = SparseSGD(
        [RecordingEmbedding(client), RecordingEmbedding(client, fusion=False)],
        lr=0.01,
        keep_ratio=0.5,
    )

    with pytest.raises(NotImplementedError, match="Embedding Fusion"):
        optimizer.step()
    assert client.calls == []


def test_strict_topk_rejects_rdma_without_submitting():
    client = RecordingKVClient(backend="rdma")
    optimizer = SparseSGD(
        [RecordingEmbedding(client)], lr=0.01, keep_ratio=0.5
    )

    with pytest.raises(NotImplementedError, match="rdma"):
        optimizer.step()
    assert client.calls == []


def test_strict_topk_rejects_unsupported_fast_path():
    client = RecordingKVClient()
    optimizer = SparseSGD(
        [RecordingEmbedding(client, fast_path="unknown_backend")],
        lr=0.01,
        keep_ratio=0.5,
    )

    with pytest.raises(NotImplementedError, match="fast path"):
        optimizer.step()
    assert client.calls == []


def test_disabled_topk_keeps_rdma_original_update_behavior():
    client = RecordingKVClient(backend="rdma")
    optimizer = SparseSGD(
        [RecordingEmbedding(client, fusion=False)], lr=0.01
    )

    optimizer.step()

    assert len(client.calls) == 1
    _, ids, grads = client.calls[0]
    torch.testing.assert_close(ids, torch.tensor([10, 20, 10, 30]))
    assert grads.shape == (4, 2)


@pytest.mark.parametrize("missing", [True, False])
def test_strict_topk_rejects_missing_or_noncallable_resolver_before_updates(missing):
    client = RecordingKVClient()
    supported = RecordingEmbedding(client)
    unsupported = RecordingEmbedding(client)
    if missing:
        # Keep the same public attributes but omit the class's resolver method.
        unsupported = SimpleNamespace(**vars(unsupported))
    else:
        unsupported.resolve_fast_path_backend = None
    optimizer = SparseSGD([supported, unsupported], lr=0.01, keep_ratio=0.5)

    with pytest.raises(NotImplementedError, match="resolve_fast_path_backend"):
        optimizer.step()

    assert client.calls == []
    assert len(supported._trace) == 1
    assert len(unsupported._trace) == 1


def test_validate_config_does_not_submit_updates_or_consume_traces():
    client = RecordingKVClient()
    embedding = RecordingEmbedding(client)
    optimizer = SparseSGD([embedding], lr=0.01, keep_ratio=0.5)

    optimizer.validate_config()

    assert client.calls == []
    assert len(embedding._trace) == 1
    optimizer.step()
    assert len(client.calls) == 1


@pytest.mark.parametrize("method,shared_local", [
    ("update_async", False),
    ("wait", False),
    ("local_update_flat", False),
    ("local_update_flat", True),
])
def test_strict_topk_rejects_missing_client_methods_before_updates(method, shared_local):
    client = RecordingKVClient()
    unsupported_client = RecordingKVClient()
    unsupported = RecordingEmbedding(unsupported_client)
    if method == "local_update_flat":
        if shared_local:
            unsupported.fast_path_mode = "auto"
            unsupported._master_config = object()
            unsupported_client.is_shared_local_shm_table = lambda: True
        else:
            unsupported._fast_path = "hierkv"
    else:
        setattr(unsupported_client, method, None)
    optimizer = SparseSGD(
        [RecordingEmbedding(client), unsupported], lr=0.01, keep_ratio=0.5
    )

    with pytest.raises(NotImplementedError, match=method):
        optimizer.step()

    assert client.calls == []


def test_disabled_topk_validation_does_not_query_backend():
    client = RecordingKVClient()
    embedding = RecordingEmbedding(client, fusion=False)

    def forbidden():
        pytest.fail("Disabled Top-K must not require backend introspection")

    client.current_ps_backend = forbidden
    embedding.resolve_fast_path_backend = forbidden
    SparseSGD([embedding], lr=0.01).validate_config()
