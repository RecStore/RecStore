#!/usr/bin/env python3

import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ps_server_helpers import find_ps_server_launcher_cli


class PSServerRunner:
    """Thin Python wrapper around ps_server_launcher_cli serve."""

    def __init__(
        self,
        server_path: str = "./build/bin/ps_server",
        config_path: Optional[str] = None,
        log_dir: str = "/tmp/recstore_ps",
        timeout: int = 120,
        num_shards: int = 2,
        verbose: bool = False,
        startup_delay: float = 2.0,
        launcher_cli: Optional[str] = None,
    ):
        self.server_path = Path(server_path)
        self.config_path = Path(config_path) if config_path else None
        self.log_dir = Path(log_dir)
        self.timeout = timeout
        self.num_shards = num_shards
        self.verbose = verbose
        self.startup_delay = startup_delay
        self.launcher_cli = launcher_cli or find_ps_server_launcher_cli()
        self.process: Optional[subprocess.Popen] = None
        self._skip_reason: Optional[str] = None

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _serve_command(self) -> list[str]:
        cmd = [
            self.launcher_cli,
            "serve",
            "--server-path",
            str(self.server_path),
            "--log-dir",
            str(self.log_dir),
            "--timeout",
            str(self.timeout),
            "--num-shards",
            str(self.num_shards),
            "--startup-delay-ms",
            str(int(self.startup_delay * 1000)),
        ]
        if self.config_path:
            cmd.extend(["--config", str(self.config_path)])
        if self.verbose:
            cmd.append("--verbose")
        return cmd

    def _drain_output(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            if self.verbose:
                print(line.rstrip("\n"), flush=True)

    def start(self) -> bool:
        if not self.server_path.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_path}")

        self.process = subprocess.Popen(
            self._serve_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert self.process.stdout is not None
        ready = False
        for line in self.process.stdout:
            line = line.rstrip("\n")
            if self.verbose and line:
                print(line, flush=True)

            if line == "READY":
                ready = True
                break
            if line.startswith("SKIP\t"):
                self._skip_reason = line.split("\t", 1)[1]
                self.process.wait(timeout=5)
                self.process = None
                return True

            if self.process.poll() is not None:
                break

        if not ready:
            stderr_tail = ""
            if self.process.stdout:
                stderr_tail = self.process.stdout.read()
            rc = self.process.returncode
            if rc is None:
                self.process.wait(timeout=5)
                rc = self.process.returncode
            self.stop()
            raise RuntimeError(
                "ps_server_launcher_cli serve failed "
                f"(rc={rc}): {stderr_tail.strip()}"
            )

        threading.Thread(target=self._drain_output, daemon=True).start()
        return True

    def stop(self):
        if self.process is None:
            return

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @contextmanager
    def run(self):
        success = self.start()
        if not success:
            raise RuntimeError("Failed to start PS Server")

        try:
            yield self
        finally:
            self.stop()


@contextmanager
def ps_server_context(
    server_path: str = "./build/bin/ps_server",
    config_path: Optional[str] = None,
    log_dir: str = "/tmp/recstore_ps",
    timeout: int = 120,
    num_shards: int = 2,
    verbose: bool = False,
    startup_delay: float = 2.0,
):
    runner = PSServerRunner(
        server_path=server_path,
        config_path=config_path,
        log_dir=log_dir,
        timeout=timeout,
        num_shards=num_shards,
        verbose=verbose,
        startup_delay=startup_delay,
    )

    with runner.run():
        yield runner
