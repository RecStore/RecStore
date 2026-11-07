
from types import SimpleNamespace

import pytest
import torch
import recstore.optimizer as optimizer_module
from recstore.optimizer import SparseSGD


class FakeDistributedClient:
    def __init__(self):
        self.updates = []

    def current_ps_backend(self):
        return "hierkv"

    def local_update_flat(self, name, ids, grads, learning_rate=0.0):
        self.updates.append((
            name,
            ids.detach().clone(),
            grads.detach().clone(),
        ))


class FakeDistributedEmbedding:
    def __init__(self, client):
        self.kv_client = client
        self.fast_path_mode = "auto"
        self._enable_fusion = True
        self._master_config = None
        self._config_names = {"feature": "master"}

        self._trace = [
            (
                "master",
                torch.tensor([10, 10, 11, 12]),
                torch.tensor([
                    [1.0, 2.0],
                    [5.0, 6.0],
                    [0.0, 2.0],
                    [0.1, 0.0],
                ]),
            )
        ]

    def resolve_fast_path_backend(self):
        return "hierkv"

    def reset_trace(self):
        self._trace.clear()


def test_rank_local_topk_before_exchange(monkeypatch):
    client = FakeDistributedClient()
    embedding = FakeDistributedEmbedding(client)

    # Represent rank zero in a two-rank process group.
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda: 0
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda: 2
    )

    captured = {}

    def fake_exchange(payload, world_size, backend):
        # Capture the payload before exchange.
        assert world_size == 2
        captured["ids"] = payload.fused_ids.clone()
        captured["grads"] = payload.grads.clone()
        captured["destinations"] = payload.destination_ranks.clone()

        # Return only the rows owned by rank zero.
        mask = payload.destination_ranks == 0
        return [
            SimpleNamespace(
                fused_ids=payload.fused_ids[mask],
                grads=payload.grads[mask],
            )
        ]

    monkeypatch.setattr(
        optimizer_module,
        "exchange_sparse_grads",
        fake_exchange,
    )

    optimizer = SparseSGD(
        params=[embedding],
        lr=0.01,
        keep_ratio=0.5,
    )

    optimizer.step()

    # Top-K must run before exchange.
    torch.testing.assert_close(
        captured["ids"],
        torch.tensor([10, 11]),
    )
    torch.testing.assert_close(
        captured["grads"],
        torch.tensor([[6.0, 8.0], [0.0, 2.0]]),
    )

    # Check destination ranks for the selected IDs.
    torch.testing.assert_close(
        captured["destinations"],
        torch.tensor([0, 1]),
    )

    # Rank zero updates only its owned ID 10.
    assert len(client.updates) == 1
    name, ids, grads = client.updates[0]

    assert name == "master"
    torch.testing.assert_close(ids, torch.tensor([10]))
    torch.testing.assert_close(
        grads,
        torch.tensor([[6.0, 8.0]]),
    )


def test_distributed_topk_requires_fusion(monkeypatch):
    client = FakeDistributedClient()
    embedding = FakeDistributedEmbedding(client)

    # Disable fusion to exercise strict rejection.
    embedding._enable_fusion = False

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    optimizer = SparseSGD(
        params=[embedding],
        lr=0.01,
        keep_ratio=0.5,
    )

    with pytest.raises(NotImplementedError, match="Embedding Fusion"):
        optimizer.step()
