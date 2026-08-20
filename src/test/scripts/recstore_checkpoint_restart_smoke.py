#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torch
from torchrec.optim import RowWiseAdagrad

from recstore.KVClient import RecStoreClient


LIBRARY_PATH = "/workspace/run/build/recstore-plugin/lib/lib_recstore_ops.so"
CONFIG_PATH = "/workspace/run/configs/recstore-hstu-smoke.json"
TABLE_NAME = "recstore_checkpoint_restart_dim512"
NUM_EMBEDDINGS = 16
EMBEDDING_DIM = 512
LEARNING_RATE = 0.001
EPSILON = 1e-8


def _metadata() -> dict:
    return {
        "checkpoint_id": "rowwise-restart-smoke-v1",
        "identity": {
            "schema_version": 1,
            "test": "rowwise-accumulator-restart",
            "table": {
                "name": TABLE_NAME,
                "num_embeddings": NUM_EMBEDDINGS,
                "embedding_dim": EMBEDDING_DIM,
            },
            "optimizer": {
                "type": "RowWiseAdagrad",
                "learning_rate": LEARNING_RATE,
                "epsilon": EPSILON,
            },
        },
    }


def _gradient(a: float, b: float, rows: int) -> torch.Tensor:
    return torch.tensor([a, b], dtype=torch.float32).repeat(
        rows, EMBEDDING_DIM // 2
    )


def _step_reference(
    parameter: torch.nn.Parameter,
    optimizer: RowWiseAdagrad,
    gradient: torch.Tensor,
) -> torch.Tensor:
    parameter.grad = gradient.clone()
    optimizer.step()
    optimizer.zero_grad()
    return parameter.detach().clone()


def _new_client() -> RecStoreClient:
    active_config = os.environ.get("RECSTORE_CONFIG")
    if active_config != CONFIG_PATH:
        raise RuntimeError(
            f"RECSTORE_CONFIG must be {CONFIG_PATH}, got {active_config!r}"
        )
    client = RecStoreClient(LIBRARY_PATH)
    client.init_data(
        name=TABLE_NAME,
        shape=(NUM_EMBEDDINGS, EMBEDDING_DIM),
        dtype=torch.float32,
        initialize_values=False,
    )
    return client


def save_phase(checkpoint_path: Path, state_path: Path) -> None:
    client = _new_client()
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    gradients = (
        _gradient(3.0, 4.0, ids.numel()),
        _gradient(1.0, -2.0, ids.numel()),
        _gradient(-0.5, 2.5, ids.numel()),
    )

    before = client.pull(TABLE_NAME, ids)
    reference = torch.nn.Parameter(before.clone())
    optimizer = RowWiseAdagrad(
        [reference], lr=LEARNING_RATE, eps=EPSILON
    )

    for step, gradient in enumerate(gradients[:2], start=1):
        client.update(TABLE_NAME, ids, gradient)
        expected = _step_reference(reference, optimizer, gradient)
        actual = client.pull(TABLE_NAME, ids)
        if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-7):
            raise AssertionError(f"pre-checkpoint update {step} mismatched TorchRec")

    expected_after_two = reference.detach().clone()
    expected_after_three = _step_reference(reference, optimizer, gradients[2])

    reset_parameter = torch.nn.Parameter(expected_after_two.clone())
    reset_optimizer = RowWiseAdagrad(
        [reset_parameter], lr=LEARNING_RATE, eps=EPSILON
    )
    reset_after_three = _step_reference(
        reset_parameter, reset_optimizer, gradients[2]
    )
    if torch.allclose(
        expected_after_three, reset_after_three, rtol=1e-6, atol=1e-7
    ):
        raise AssertionError("third gradient does not distinguish restored state")

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": ids,
            "third_gradient": gradients[2],
            "expected_after_two": expected_after_two,
            "expected_after_three": expected_after_three,
            "reset_after_three": reset_after_three,
        },
        state_path,
    )
    client.save_checkpoint(str(checkpoint_path), _metadata())
    print("SAVE_PHASE=PASS")


def load_phase(checkpoint_path: Path, state_path: Path) -> None:
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    client = _new_client()
    client.load_checkpoint(str(checkpoint_path), _metadata())

    restored = client.pull(TABLE_NAME, state["ids"])
    if not torch.allclose(
        restored, state["expected_after_two"], rtol=1e-6, atol=1e-7
    ):
        raise AssertionError("restored embedding rows differ from saved rows")

    client.update(TABLE_NAME, state["ids"], state["third_gradient"])
    after_third = client.pull(TABLE_NAME, state["ids"])
    if not torch.allclose(
        after_third, state["expected_after_three"], rtol=1e-6, atol=1e-7
    ):
        raise AssertionError("restored RowWiseAdagrad state is not continuous")
    if torch.allclose(
        after_third, state["reset_after_three"], rtol=1e-6, atol=1e-7
    ):
        raise AssertionError("RowWiseAdagrad accumulator was reset after restart")

    max_error = (after_third - state["expected_after_three"]).abs().max().item()
    print(f"max_abs_error={max_error:.10g}")
    print("ACCUMULATOR_RESTORED=True")
    print("LOAD_PHASE=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("save", "load"))
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("state_path", type=Path)
    args = parser.parse_args()
    if args.phase == "save":
        save_phase(args.checkpoint_path, args.state_path)
    else:
        load_phase(args.checkpoint_path, args.state_path)


if __name__ == "__main__":
    main()
