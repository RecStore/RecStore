from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from .common import ROOT


def _run(cmd: list[str], *, cwd: Path = ROOT, log_path: Path | None = None, dry_run: bool = False) -> int:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("$ " + format_command(cmd) + "\n")
    if dry_run:
        return 0
    start = time.time()
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as sink:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False,
            )
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[exit_code] {proc.returncode}\n[elapsed_s] {time.time() - start:.3f}\n")
    return int(proc.returncode)


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def wrap_remote_command(
    cmd: list[str],
    host: str,
    *,
    cwd: Path,
    ssh_port: int = 22,
) -> list[str]:
    remote = "cd {cwd} && {cmd}".format(
        cwd=shlex.quote(str(cwd)),
        cmd=" ".join(shlex.quote(part) for part in cmd),
    )
    ssh_cmd = ["ssh"]
    if ssh_port != 22:
        ssh_cmd.extend(["-p", str(ssh_port)])
    ssh_cmd.extend([host.strip(), remote])
    return ssh_cmd
