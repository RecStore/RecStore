#!/usr/bin/env python3

import os
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.recstore_config_path import find_recstore_config_path


RDMA_SKIP_EXIT_CODE = 77
BUILD_BIN = REPO_ROOT / "build" / "bin"


def find_ps_server_launcher_cli():
    """Return ps_server_launcher_cli under build/bin."""
    return str(BUILD_BIN / "ps_server_launcher_cli")


def find_ps_server_binary():
    """Return ps_server under build/bin."""
    return str(BUILD_BIN / "ps_server")


def _launcher_env(config_path=None):
    env = os.environ.copy()
    if config_path:
        env["RECSTORE_CONFIG"] = str(config_path)
    return env


def run_launcher_decision(config_path=None):
    """Run C++ launch decision and return parsed JSON."""
    cmd = [find_ps_server_launcher_cli(), "decision"]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_launcher_env(config_path),
        check=False,
    )
    if proc.returncode not in (0, 2):
        raise RuntimeError(
            "ps_server_launcher_cli decision failed: "
            f"rc={proc.returncode} stderr={proc.stderr.strip()}"
        )

    decision = json.loads(proc.stdout)
    if proc.returncode == 2 and not decision.get("should_fail"):
        decision["should_fail"] = True
    return decision


def find_config_file():
    """Find the active RecStore config file."""
    config_path = find_recstore_config_path(os.getcwd())
    return str(config_path) if config_path is not None else None


def load_config():
    """Load the active RecStore config file."""
    config_path = find_config_file()
    if not config_path:
        return None, {}

    with open(config_path, "r") as f:
        return config_path, json.load(f)


def get_backend_type():
    """Return the configured backend type for the current test run."""
    _config_path, config = load_config()
    cache_ps = config.get("cache_ps", {})
    return str(cache_ps.get("ps_type", "GRPC")).upper()


def get_rdma_runner_config():
    """Extract the RDMA runner settings needed by the PetPS test harness."""
    _config_path, config = load_config()
    cache_ps = config.get("cache_ps", {})
    dist_client = config.get("distributed_client", {})
    base_kv = cache_ps.get("base_kv_config", {})
    return {
        "num_servers": int(
            dist_client.get("num_shards", cache_ps.get("num_shards", 1))
        ),
        "value_size": int(
            base_kv.get("value", {}).get(
                "default_value_size_hint", base_kv.get("value_size", 512)
            )
        ),
        "max_kv_num_per_request": int(dist_client.get("max_keys_per_request", 64)),
    }


def get_rdma_skip_reason():
    """Return skip reason when RDMA verbs devices are not available."""
    rdma_device_dir = "/dev/infiniband"
    if not os.path.isdir(rdma_device_dir):
        return f"RDMA verbs device directory is unavailable: {rdma_device_dir}"

    uverbs_devices = sorted(glob.glob(os.path.join(rdma_device_dir, "uverbs*")))
    if not uverbs_devices:
        return f"RDMA verbs devices are unavailable under {rdma_device_dir}"

    return None


def get_ports_from_config():
    """Extract ports from recstore_config.json via the C++ launcher."""
    decision = run_launcher_decision()
    return decision.get("configured_ports") or [15000, 15001, 15002, 15003]


def check_ps_server_running(ports=None):
    """Check if ps_server is running by checking if ports are open."""
    decision = run_launcher_decision()
    configured_ports = decision.get("configured_ports") or []
    open_set = set(decision.get("open_ports") or [])
    check_ports = ports if ports is not None else configured_ports
    open_ports = [port for port in check_ports if port in open_set]

    if open_ports:
        return True, open_ports
    return False, []


def should_skip_server_start():
    """Determine if we should skip starting ps_server."""
    decision = run_launcher_decision()

    if decision.get("should_fail"):
        raise RuntimeError(decision.get("reason", "ps_server launch decision failed"))

    if not decision.get("should_start"):
        reason = decision.get("reason") or "skip"
        open_ports = decision.get("open_ports") or []
        if reason in ("already_running", "ci_reuse_running"):
            return True, f"{reason}:{open_ports}"
        if reason == "NO_PS_SERVER":
            return True, "NO_PS_SERVER"
        return True, reason

    is_ci = os.environ.get("CI") == "true" or os.environ.get(
        "GITHUB_ACTIONS"
    ) == "true"
    if is_ci and decision.get("reason") == "ci_server_not_ready":
        configured_ports = decision.get("configured_ports") or []
        open_ports = decision.get("open_ports") or []
        return False, (
            f"ci_server_not_ready: expected={configured_ports}, open={open_ports}"
        )

    return False, None


def get_server_config():
    """Get server configuration from environment."""
    return {
        "server_path": find_ps_server_binary(),
        "launcher_cli": find_ps_server_launcher_cli(),
        "config_path": os.environ.get("RECSTORE_CONFIG"),
        "log_dir": os.environ.get("PS_LOG_DIR", "/tmp/recstore_ps"),
        "timeout": int(os.environ.get("PS_TIMEOUT", "60")),
        "num_shards": int(os.environ.get("PS_NUM_SHARDS", "2")),
    }
