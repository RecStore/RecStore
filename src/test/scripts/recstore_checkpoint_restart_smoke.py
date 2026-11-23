#!/usr/bin/env python3
"""Save a committed PS checkpoint, kill the PS, and let it auto-restore.

Phase layout (``--phase``):

- ``driver`` (default): writes a config with ``cache_ps.checkpoint``, starts
  ``ps_server``, runs ``save``, SIGKILLs it, restarts the same config and runs
  ``verify``. The second start must restore from the manifest without any
  explicit ``load_checkpoint`` call.
- ``save``: initialize rows in shard files, apply three SGD steps, save a
  checkpoint through ``recstore.checkpoint.save_sparse_checkpoint`` and record
  the expected rows.
- ``verify``: pull the rows (they must already be restored) and apply one more
  SGD step to prove the optimizer state came back too.

The phases run as separate processes because ``recstore`` binds its PS config
at import time.

Example:

    PYTHONPATH=src/python/pytorch python3 src/test/scripts/recstore_checkpoint_restart_smoke.py \
        --run-dir results/ps_ha_restore_11221250
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
TABLE = "t_ha_restore"
NUM_EMBEDDINGS = 64
DIM = 8
LR = 0.01
IDENTITY = {
    "schema_version": 1,
    "ps_type": "GRPC",
    "tables": [
        {"name": TABLE, "num_embeddings": NUM_EMBEDDINGS, "embedding_dim": DIM}
    ],
}


def _client(initial: torch.Tensor | None = None):
    from recstore.KVClient import RecStoreClient

    client = RecStoreClient()
    full = torch.zeros(NUM_EMBEDDINGS, DIM, dtype=torch.float32)
    if initial is not None:
        full[1:5] = initial
    client.init_data(
        name=TABLE,
        shape=(NUM_EMBEDDINGS, DIM),
        dtype=torch.float32,
        initialize_values=initial is not None,
        init_func=lambda shape, dtype: full,
    )
    return client


def _rows(scale: float) -> torch.Tensor:
    rows = torch.arange(4 * DIM, dtype=torch.float32).reshape(4, DIM)
    return (rows + scale) * 0.125


def save_phase(run_dir: Path) -> None:
    from recstore.checkpoint import latest_committed, save_sparse_checkpoint

    keys = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    values = _rows(1.0)
    client = _client(values)
    assert torch.equal(client.pull(TABLE, keys), values)

    step_grad = _rows(3.0)
    for _ in range(3):
        client.update(TABLE, keys, step_grad)
    expected = values - LR * 3 * step_grad
    max_error = (client.pull(TABLE, keys) - expected).abs().max().item()
    assert max_error < 1e-6, max_error

    checkpoint_root = run_dir / "ckpt"
    checkpoint = save_sparse_checkpoint(
        client, checkpoint_root, step=5, identity=IDENTITY, keep=2
    )
    manifest = latest_committed(checkpoint_root)
    torch.save(
        {"keys": keys, "expected": expected, "grad": step_grad},
        run_dir / "state.pt",
    )
    print(f"SAVE_PHASE=PASS checkpoint={checkpoint.checkpoint_id}")
    print(f"MANIFEST_STEP={manifest['step']}")
    print(f"MANIFEST_FILES={[item['name'] for item in manifest['files']]}")


def verify_phase(run_dir: Path) -> None:
    client = _client()
    state = torch.load(run_dir / "state.pt", weights_only=True)
    keys = state["keys"]
    restored = client.pull(TABLE, keys)
    max_error = (restored - state["expected"]).abs().max().item()
    assert max_error < 1e-6, f"restored rows differ: max_abs={max_error}"
    print(f"RESTORED_MAX_ABS={max_error:.3e}")

    client.update(TABLE, keys, state["grad"])
    expected = state["expected"] - LR * state["grad"]
    max_error = (client.pull(TABLE, keys) - expected).abs().max().item()
    assert max_error < 1e-6, f"post-restore update differs: max_abs={max_error}"
    print(f"POST_UPDATE_MAX_ABS={max_error:.3e}")
    print("VERIFY_PHASE=PASS")


def _write_config(base: Path, run_dir: Path) -> Path:
    config = json.loads(base.read_text())
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
