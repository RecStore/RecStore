#!/usr/bin/env python3
"""Exercise AdamW with two namespaced tables across a PS checkpoint restart.

This is the regression test for RecStore issue #400: AdamW used to tag its
parameter, moments and step with the role only, so two tables on one PS shared a
key space and silently overwrote each other. The tags now carry ``table_id``
(``MakeTensorTag(table_id, role, 4)``), and the client encodes its keys with the
tag the server returned from table registration.

Both tables here use non-zero ``table_id`` (1 and 2) so the tag round trip is
actually exercised: a tag of 0 would leave the keys byte-identical to the old
behaviour.

Phase layout (``--phase``):

- ``driver`` (default): writes a config with AdamW plus ``cache_ps.checkpoint``,
  starts ``ps_server``, runs ``save``, SIGKILLs it, restarts the same config and
  runs ``verify``.
- ``save``: register two tables with distinct ``table_id``, write distinct rows
  under the same ids, apply three AdamW updates and check the rows against an
  independent implementation of the update rule, then save a checkpoint.
- ``verify``: pull the restored rows and apply one more update, so a restored
  step counter (bias correction at step 4, not 1) is required to pass.

Example:

    PYTHONPATH=src/python/pytorch python3 \
        src/test/scripts/adamw_multi_table_restart_smoke.py \
        --run-dir results/adamw_multi_table_restart_11241005
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

NUM_EMBEDDINGS = 64
DIM = 8
KEYS = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

# Kept in sync with the server-side AdamW constructor arguments below.
LR = 0.05
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-8
WEIGHT_DECAY = 0.01
TENSORS_PER_TABLE = 4

# The default config uses 15000/15001, which the SGD restart smoke test also
# binds; move to a private pair so both can run in one ctest invocation.
PORT_MAP = {15000: 15200, 15001: 15201}

TABLES = {
    # table_id 0 keeps tag 0, so its keys stay byte-identical to the old
    # untagged layout: deployments that never saw a tag keep working.
    "t_adamw_zero": {"table_id": 0, "initial": 0.5, "grad_scale": 0.75},
    "t_adamw_a": {"table_id": 1, "initial": 1.0, "grad_scale": 1.0},
    "t_adamw_b": {"table_id": 2, "initial": -1.0, "grad_scale": -0.5},
}

IDENTITY = {
    "schema_version": 1,
    "ps_type": "GRPC",
    "tables": [
        {
            "name": name,
            "table_id": meta["table_id"],
            "num_embeddings": NUM_EMBEDDINGS,
            "embedding_dim": DIM,
        }
        for name, meta in TABLES.items()
    ],
}


def _initial_rows(scale: float) -> torch.Tensor:
    rows = torch.arange(NUM_EMBEDDINGS * DIM, dtype=torch.float32).reshape(
        NUM_EMBEDDINGS, DIM
    )
    return (rows.remainder(17).to(torch.float64) + 1.0) * scale * 0.125


def _grads(scale: float, update_index: int) -> torch.Tensor:
    rows = torch.arange(len(KEYS) * DIM, dtype=torch.float32).reshape(len(KEYS), DIM)
    return ((rows + 1.0 + update_index) * scale * 0.25).to(torch.float64)


def _client() -> object:
    from recstore.KVClient import RecStoreClient

    return RecStoreClient()


def _register(client, name: str, initial: bool):
    meta = TABLES[name]
    full = _initial_rows(meta["initial"])
    client.init_data(
        name=name,
        shape=(NUM_EMBEDDINGS, DIM),
        dtype=torch.float32,
        initialize_values=initial,
        backend_table_id=meta["table_id"],
        init_func=lambda shape, dtype: full.float() if initial else None,
    )


def _adamw(param, moment1, moment2, step, grads):
    """Mirror of AdamW::UpdateRows, in float64."""
    for index, grad in enumerate(grads):
        step += 1
        moment1 = BETA1 * moment1 + (1.0 - BETA1) * grad
        moment2 = BETA2 * moment2 + (1.0 - BETA2) * grad * grad
        m_hat = moment1 / (1.0 - BETA1**step)
        v_hat = moment2 / (1.0 - BETA2**step)
        param = (1.0 - LR * WEIGHT_DECAY) * param - LR * m_hat / (
            torch.sqrt(v_hat) + EPS
        )
    return param, moment1, moment2, step


def _expected(name: str, steps: int):
    meta = TABLES[name]
    param = _initial_rows(meta["initial"])[KEYS].clone()
    moment1 = torch.zeros_like(param)
    moment2 = torch.zeros_like(param)
    step = 0
    grads = []
    for index in range(steps):
        grad = _grads(meta["grad_scale"], index)
        grads.append(grad)
        param, moment1, moment2, step = _adamw(
            param, moment1, moment2, step, [grad]
        )
    return param.float(), step, grads


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.to(torch.float64) - expected.to(torch.float64)).abs().max().item()


def save_phase(run_dir: Path) -> None:
    from recstore.checkpoint import latest_committed, save_sparse_checkpoint

    client = _client()
    for name in TABLES:
        _register(client, name, initial=True)
        table_id, tag = client.get_embedding_table_namespace(name)
        wanted = TABLES[name]["table_id"]
        assert table_id == wanted, f"{name}: table_id={table_id} wanted {wanted}"
        assert tag == TENSORS_PER_TABLE * wanted, (
            f"{name}: tag={tag} wanted {TENSORS_PER_TABLE * wanted}"
        )
        print(f"NAMESPACE={name} table_id={table_id} tag={tag}")

    expected = {}
    grads = {}
    for name, meta in TABLES.items():
        initial = _initial_rows(meta["initial"])[KEYS].float()
        assert torch.allclose(client.pull(name, KEYS), initial, atol=1e-6), (
            f"{name}: initial rows did not round trip"
        )
        for index in range(3):
            grad = _grads(meta["grad_scale"], index).float()
            client.update(name, KEYS, grad)
            grads.setdefault(name, []).append(grad)
        expected[name], step, _ = _expected(name, 3)
        assert step == 3, step
        error = _max_abs(client.pull(name, KEYS), expected[name])
        assert error < 1e-5, f"{name}: after 3 updates max_abs={error}"
        print(f"AFTER_3_UPDATES={name} max_abs={error:.3e}")

    names = list(TABLES)
    first, second = names[0], names[1]
    overlap = _max_abs(
        client.pull(first, KEYS), client.pull(second, KEYS)
    )
    assert overlap > 1e-3, f"tables {first}/{second} hold identical rows"
    print(f"TABLE_GAP={overlap:.3e}")

    checkpoint = save_sparse_checkpoint(
        client, run_dir / "ckpt", step=5, identity=IDENTITY, keep=2
    )
    manifest = latest_committed(run_dir / "ckpt")
    torch.save(
        {"expected": expected, "grads": grads},
        run_dir / "state.pt",
    )
    print(f"SAVE_PHASE=PASS checkpoint={checkpoint.checkpoint_id}")
    print(f"MANIFEST_STEP={manifest['step']}")


def verify_phase(run_dir: Path) -> None:
    client = _client()
    for name in TABLES:
        _register(client, name, initial=False)

    state = torch.load(run_dir / "state.pt", weights_only=True)
    for name in TABLES:
        restored = client.pull(name, KEYS)
        error = _max_abs(restored, state["expected"][name])
        assert error < 1e-5, f"{name}: restored rows differ max_abs={error}"
        print(f"RESTORED={name} max_abs={error:.3e}")

    # The fourth update only matches when the restored step advanced 3 -> 4;
    # a lost step counter would use the step-1 bias correction instead.
    for name in TABLES:
        client.update(name, KEYS, _grads(TABLES[name]["grad_scale"], 3).float())
        expected, _, _ = _expected(name, 4)
        error = _max_abs(client.pull(name, KEYS), expected)
        assert error < 1e-5, f"{name}: post-restore update differs max_abs={error}"
        print(f"POST_RESTORE_STEP4={name} max_abs={error:.3e}")

    print("VERIFY_PHASE=PASS")


def _remap_ports(node):
    if isinstance(node, dict):
        return {key: _remap_ports(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_remap_ports(item) for item in node]
    if isinstance(node, int) and not isinstance(node, bool):
        return PORT_MAP.get(node, node)
    return node


def _write_config(base: Path, run_dir: Path) -> Path:
    config = _remap_ports(json.loads(base.read_text()))
    config["cache_ps"]["optimizer"] = {
        "type": "AdamW",
        "learning_rate": LR,
        "beta1": BETA1,
        "beta2": BETA2,
        "epsilon": EPS,
        "weight_decay": WEIGHT_DECAY,
    }
    config["cache_ps"]["checkpoint"] = {
        "restore_dir": str(run_dir / "ckpt"),
        "on_missing": "start_empty",
    }
    # Keep the run off the shared /dev/shm names another run may still hold.
    value = config["cache_ps"].get("base_kv_config", {}).get("value", {})
    if "path" in value:
        value["path"] = str(run_dir / "kv" / "value")
    target = run_dir / "config.json"
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target


def _ports(config: Path) -> list[int]:
    cache_ps = json.loads(config.read_text())["cache_ps"]
    return [int(server["port"]) for server in cache_ps.get("servers", [])]


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _start_ps(ps_binary: Path, config: Path, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("RECSTORE_CONFIG", None)
    handle = log_path.open("w")
    process = subprocess.Popen(
        [str(ps_binary), "-config_path", str(config), "-logtostderr=1"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    ports = _ports(config) or [15000]
    deadline = time.time() + 60
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ps_server exited early; see {log_path}")
        if all(_port_open(port) for port in ports):
            return process
        time.sleep(0.2)
    raise RuntimeError(f"ps_server did not listen in time; see {log_path}")


def _stop_ps(process: subprocess.Popen, sig: int) -> int:
    process.send_signal(sig)
    try:
        return process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=15)


def _run_phase(script: Path, phase: str, config: Path, run_dir: Path) -> None:
    env = dict(os.environ)
    env["RECSTORE_CONFIG"] = str(config)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src' / 'python' / 'pytorch'}:{env.get('PYTHONPATH', '')}"
    )
    subprocess.run(
        [sys.executable, str(script), "--phase", phase, "--run-dir", str(run_dir)],
        check=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def driver(base: Path, run_dir: Path, ps_binary: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _write_config(base, run_dir)
    script = Path(__file__).resolve()

    first = _start_ps(ps_binary, config, run_dir / "ps_first.log")
    try:
        _run_phase(script, "save", config, run_dir)
    finally:
        code = _stop_ps(first, signal.SIGKILL)
    print(f"FIRST_PS_EXIT={code}")

    second = _start_ps(ps_binary, config, run_dir / "ps_restored.log")
    try:
        log = (run_dir / "ps_restored.log").read_text(errors="replace")
        restored = [line for line in log.splitlines() if "Restored shard" in line]
        if not restored:
            raise AssertionError("PS log has no 'Restored shard' line")
        for line in restored:
            print("RESTORE_LOG=" + line.split("] ", 1)[-1])
        _run_phase(script, "verify", config, run_dir)
    finally:
        _stop_ps(second, signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("driver", "save", "verify"), default="driver")
    parser.add_argument(
        "--base-config", default=str(REPO_ROOT / "recstore_config.json")
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--ps-binary", default=str(REPO_ROOT / "build" / "bin" / "ps_server")
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if args.phase == "save":
        save_phase(run_dir)
    elif args.phase == "verify":
        verify_phase(run_dir)
    else:
        shutil.rmtree(run_dir / "ckpt", ignore_errors=True)
        driver(Path(args.base_config), run_dir, Path(args.ps_binary))


if __name__ == "__main__":
    main()
