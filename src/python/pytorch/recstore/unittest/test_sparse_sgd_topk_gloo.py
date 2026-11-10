
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from recstore.optimizer import SparseSGD

from recstore.single_node_exchange import (
    SparseGradPayload,
    exchange_sparse_grads,
)


class FakeKVClient:
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


class FakeEmbedding:
    def __init__(self, client, ids, grads):
        self.kv_client = client
        self.fast_path_mode = "auto"
        self._enable_fusion = True
        self._master_config = None
        self._config_names = {"feature": "master"}
        self._trace = [("master", ids, grads)]

    def resolve_fast_path_backend(self):
        return "hierkv"

    def reset_trace(self):
        self._trace.clear()


def run_rank(rank, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )

    try:
        client = FakeKVClient()

        if rank == 0:
            ids = torch.tensor([10, 10, 11, 12])
            grads = torch.tensor([
                [1.0, 2.0],
                [5.0, 6.0],
                [0.0, 2.0],
                [0.1, 0.0],
            ])
        else:
            ids = torch.tensor([10, 13, 14])
            grads = torch.tensor([
                [1.0, 1.0],
                [5.0, 0.0],
                [0.1, 0.0],
            ])

        embedding = FakeEmbedding(client, ids, grads)

        optimizer = SparseSGD(
            params=[embedding],
            lr=0.01,
            keep_ratio=0.5,
        )

        # Use the real distributed gradient exchange.
        optimizer.step()

        assert len(client.updates) == 1
        name, updated_ids, updated_grads = client.updates[0]
        assert name == "master"

        if rank == 0:
            torch.testing.assert_close(
                updated_ids,
                torch.tensor([10]),
            )
            torch.testing.assert_close(
                updated_grads,
                torch.tensor([[7.0, 9.0]]),
            )
        else:
            torch.testing.assert_close(
                updated_ids,
                torch.tensor([11, 13]),
            )
            torch.testing.assert_close(
                updated_grads,
                torch.tensor([
                    [0.0, 2.0],
                    [5.0, 0.0],
                ]),
            )

        print(f"Rank {rank}: Gloo Top-K exchange passed", flush=True)

    finally:
        dist.destroy_process_group()


def test_topk_real_gloo_two_ranks(tmp_path):
    init_file = tmp_path / "gloo_init"

    mp.spawn(
        run_rank,
        args=(f"file://{init_file}",),
        nprocs=2,
        join=True,
    )


def run_empty_owner_after_topk(rank, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        client = FakeKVClient()
        embedding = FakeEmbedding(
            client,
            torch.tensor([10, 11]),
            torch.tensor([[10.0 + rank, 0.0], [1.0, 0.0]]),
        )
        optimizer = SparseSGD([embedding], lr=0.01, keep_ratio=0.5)
        optimizer.step()

        if rank == 0:
            assert len(client.updates) == 1
            _, ids, grads = client.updates[0]
            torch.testing.assert_close(ids, torch.tensor([10]))
            torch.testing.assert_close(grads, torch.tensor([[21.0, 0.0]]))
        else:
            assert client.updates == []
        assert embedding._trace == []
        # Both ranks must still participate in a subsequent step.
        embedding._trace = [("master", torch.tensor([11]), torch.ones(1, 2))]
        optimizer.step()
        if rank == 1:
            _, ids, grads = client.updates[0]
            torch.testing.assert_close(ids, torch.tensor([11]))
            torch.testing.assert_close(grads, torch.tensor([[2.0, 2.0]]))
    finally:
        dist.destroy_process_group()


def test_topk_real_gloo_empty_owner(tmp_path):
    mp.spawn(
        run_empty_owner_after_topk,
        args=(f"file://{tmp_path / 'empty_owner_init'}",),
        nprocs=2,
        join=True,
    )




def run_empty_rank_exchange(rank, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )

    try:
        if rank == 0:
            ids = torch.tensor([10, 11], dtype=torch.int64)
            grads = torch.tensor([
                [1.0, 2.0],
                [3.0, 4.0],
            ], dtype=torch.float32)
        else:
            # Rank one has no outgoing gradients.
            ids = torch.empty((0,), dtype=torch.int64)
            grads = torch.empty((0, 2), dtype=torch.float32)

        payload = SparseGradPayload(
            rank=rank,
            destination_ranks=ids % 2,
            source_ranks=torch.full(
                (ids.numel(),), rank, dtype=torch.int64
            ),
            row_positions=torch.arange(
                ids.numel(), dtype=torch.int64
            ),
            fused_ids=ids,
            grads=grads,
        )

        # Exercise real communication.
        received = exchange_sparse_grads(
            payload,
            world_size=2,
            backend=dist,
        )

        received_ids = torch.cat([
            p.fused_ids for p in received
            if p.fused_ids.numel() > 0
        ])
        received_grads = torch.cat([
            p.grads for p in received
            if p.fused_ids.numel() > 0
        ])

        if rank == 0:
            expected_ids = torch.tensor([10])
            expected_grads = torch.tensor([[1.0, 2.0]])
        else:
            expected_ids = torch.tensor([11])
            expected_grads = torch.tensor([[3.0, 4.0]])

        torch.testing.assert_close(received_ids, expected_ids)
        torch.testing.assert_close(received_grads, expected_grads)

        print(f"Rank {rank}: empty-rank exchange passed", flush=True)

    finally:
        dist.destroy_process_group()


def test_sparse_grad_exchange_one_rank_empty(tmp_path):
    init_file = tmp_path / "empty_rank_init"

    mp.spawn(
        run_empty_rank_exchange,
        args=(f"file://{init_file}",),
        nprocs=2,
        join=True,
    )
