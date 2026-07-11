#!/usr/bin/env python3

import os
import queue
import re
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _supports_put_v2_client_flags(argv):
    if not argv:
        return False
    binary_name = os.path.basename(str(argv[0]))
    return binary_name in {
        "petps_integration_test",
        "petps_multiclient_stress",
    }


class PetPSClusterRunner:
    def __init__(
        self,
        server_path="./build/bin/petps_server",
        config_path="./recstore_config.json",
        num_servers=1,
        num_clients=1,
        thread_num=1,
        value_size=16,
        max_kv_num_per_request=64,
        timeout=60,
        startup_delay=2.0,
        log_dir="/tmp",
        verbose=False,
        status_refresh_interval=2.0,
        show_status_logs=True,
        show_control_plane_logs=True,
        rdma_namespace="auto",
        rdma_control_plane_host="127.0.0.1",
        rdma_control_plane_port=None,
        rdma_per_thread_response_limit_bytes=None,
        rdma_client_receive_arena_bytes=None,
        rdma_put_protocol_version=None,
        rdma_put_v2_transfer_mode=None,
        rdma_put_v2_push_slot_bytes=None,
        rdma_put_v2_push_slots_per_client=None,
        rdma_put_v2_push_region_offset=None,
        rdma_put_client_send_arena_bytes=None,
        rdma_put_server_scratch_bytes=None,
        rdma_rc_qps_per_client_per_shard=None,
        rdma_rc_slots_per_qp=None,
        rdma_rc_profile_interval_ms=None,
        rdma_rc_server_coroutines_per_thread=None,
        rdma_rc_server_get_workers=None,
        logical_clients_per_process=1,
        rdma_rc_inline_bytes=None,
        rdma_rc_client_numa_id=None,
        rdma_rc_server_numa_id=None,
        rdma_rc_fake_get_mode=None,
        rdma_rc_skip_client_copy=None,
        rdma_qps_per_client_per_shard=None,
        rdma_slots_per_qp=None,
        rdma_wait_timeout_ms=None,
        rdma_control_plane_timeout_ms=None,
        rdma_profile_interval_ms=None,
        rdma_server_coroutines_per_thread=None,
        rdma_server_get_workers=None,
        rdma_inline_bytes=None,
        rdma_client_numa_id=None,
        rdma_client_numa_ids=None,
        rdma_server_numa_id=None,
        rdma_server_bind_core_offset=None,
        rdma_client_bind_core_offset=None,
        rdma_client_bind_core_stride=None,
        rdma_fake_get_mode=None,
        rdma_skip_client_copy=None,
        validate_routing=False,
        server_command_wrapper=None,
    ):
        self.server_path = Path(server_path)
        if not self.server_path.is_absolute():
            self.server_path = (REPO_ROOT / self.server_path).resolve()

        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (REPO_ROOT / self.config_path).resolve()
        self.num_servers = num_servers
        self.num_clients = num_clients
        self.thread_num = thread_num
        self.logical_clients_per_process = max(1, logical_clients_per_process)
        self.logical_num_clients = self.num_clients * self.logical_clients_per_process
        self.value_size = value_size
        self.max_kv_num_per_request = max_kv_num_per_request
        self.timeout = timeout
        self.startup_delay = startup_delay
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.status_refresh_interval = status_refresh_interval
        self.show_status_logs = show_status_logs
        self.show_control_plane_logs = show_control_plane_logs
        self.rdma_control_plane_host = rdma_control_plane_host
        self.rdma_control_plane_port = (
            rdma_control_plane_port
            if rdma_control_plane_port is not None
            else self._allocate_local_port(rdma_control_plane_host)
        )
        if rdma_namespace == "auto":
            rdma_namespace = f"recstore-rdma-{os.getpid()}-{time.time_ns()}"
        self.rdma_namespace = rdma_namespace
        self.rdma_per_thread_response_limit_bytes = (
            rdma_per_thread_response_limit_bytes
        )
        self.rdma_client_receive_arena_bytes = rdma_client_receive_arena_bytes
        self.rdma_put_protocol_version = rdma_put_protocol_version
        self.rdma_put_v2_transfer_mode = rdma_put_v2_transfer_mode
        self.rdma_put_v2_push_slot_bytes = rdma_put_v2_push_slot_bytes
        self.rdma_put_v2_push_slots_per_client = rdma_put_v2_push_slots_per_client
        self.rdma_put_v2_push_region_offset = rdma_put_v2_push_region_offset
        self.rdma_put_client_send_arena_bytes = rdma_put_client_send_arena_bytes
        self.rdma_put_server_scratch_bytes = rdma_put_server_scratch_bytes
        self.rdma_qps_per_client_per_shard = self._coalesce_optional_values(
            "rdma_qps_per_client_per_shard",
            rdma_qps_per_client_per_shard,
            rdma_rc_qps_per_client_per_shard,
        )
        self.rdma_wait_timeout_ms = rdma_wait_timeout_ms
        self.rdma_control_plane_timeout_ms = rdma_control_plane_timeout_ms
        self.rdma_slots_per_qp = self._coalesce_optional_values(
            "rdma_slots_per_qp",
            rdma_slots_per_qp,
            rdma_rc_slots_per_qp,
        )
        self.rdma_profile_interval_ms = self._coalesce_optional_values(
            "rdma_profile_interval_ms",
            rdma_profile_interval_ms,
            rdma_rc_profile_interval_ms,
        )
        self.rdma_server_coroutines_per_thread = self._coalesce_optional_values(
            "rdma_server_coroutines_per_thread",
            rdma_server_coroutines_per_thread,
            rdma_rc_server_coroutines_per_thread,
        )
        self.rdma_server_get_workers = self._coalesce_optional_values(
            "rdma_server_get_workers",
            rdma_server_get_workers,
            rdma_rc_server_get_workers,
        )
        self.rdma_inline_bytes = self._coalesce_optional_values(
            "rdma_inline_bytes",
            rdma_inline_bytes,
            rdma_rc_inline_bytes,
        )
        self.rdma_client_numa_id = self._coalesce_optional_values(
            "rdma_client_numa_id",
            rdma_client_numa_id,
            rdma_rc_client_numa_id,
        )
        self.rdma_client_numa_ids = (
            list(rdma_client_numa_ids) if rdma_client_numa_ids is not None else None
        )
        if self.rdma_client_numa_ids is not None:
            if len(self.rdma_client_numa_ids) != self.num_clients:
                raise ValueError(
                    "rdma_client_numa_ids length must match num_clients"
                )
            if self.rdma_client_numa_id is not None:
                raise ValueError(
                    "rdma_client_numa_id and rdma_client_numa_ids are mutually exclusive"
                )
        self.rdma_server_numa_id = self._coalesce_optional_values(
            "rdma_server_numa_id",
            rdma_server_numa_id,
            rdma_rc_server_numa_id,
        )
        self.rdma_server_bind_core_offset = rdma_server_bind_core_offset
        self.rdma_client_bind_core_offset = rdma_client_bind_core_offset
        self.rdma_client_bind_core_stride = rdma_client_bind_core_stride
        if (
            self.rdma_client_bind_core_stride is not None
            and self.rdma_client_bind_core_stride < 0
        ):
            raise ValueError("rdma_client_bind_core_stride must be non-negative")
        self.rdma_fake_get_mode = self._coalesce_optional_values(
            "rdma_fake_get_mode",
            rdma_fake_get_mode,
            rdma_rc_fake_get_mode,
        )
        self.rdma_skip_client_copy = self._coalesce_optional_values(
            "rdma_skip_client_copy",
            rdma_skip_client_copy,
            rdma_rc_skip_client_copy,
        )
        self.rdma_rc_qps_per_client_per_shard = self.rdma_qps_per_client_per_shard
        self.rdma_rc_slots_per_qp = self.rdma_slots_per_qp
        self.rdma_rc_profile_interval_ms = self.rdma_profile_interval_ms
        self.rdma_rc_server_coroutines_per_thread = (
            self.rdma_server_coroutines_per_thread
        )
        self.rdma_rc_server_get_workers = self.rdma_server_get_workers
        self.rdma_rc_inline_bytes = self.rdma_inline_bytes
        self.rdma_rc_client_numa_id = self.rdma_client_numa_id
        self.rdma_rc_client_numa_ids = self.rdma_client_numa_ids
        self.rdma_rc_server_numa_id = self.rdma_server_numa_id
        self.rdma_rc_fake_get_mode = self.rdma_fake_get_mode
        self.rdma_rc_skip_client_copy = self.rdma_skip_client_copy
        self.validate_routing = validate_routing
        self.server_command_wrapper = server_command_wrapper
        self.processes = []
        self.process_logs = {}
        self.ready = set()
        self.ready_threads = {}

    @staticmethod
    def _coalesce_optional_values(name, primary, legacy):
        present = [value for value in (primary, legacy) if value is not None]
        if len(present) == 2 and present[0] != present[1]:
            raise ValueError(
                f"conflicting values provided for {name}: {primary!r} vs {legacy!r}"
            )
        return present[0] if present else None

    @staticmethod
    def _allocate_local_port(host):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return sock.getsockname()[1]

    def emit_status(self, phase, extra=""):
        if not self.show_status_logs:
            return
        running_pids = [
            str(process.pid)
            for process, _thread in self.processes
            if process.poll() is None
        ]
        detail = (
            f" phase={phase} ready={len(self.ready)}/{self.num_servers}"
            f" running_pids={','.join(running_pids) if running_pids else 'none'}"
        )
        if extra:
            detail += f" {extra}"
        print(f"[petps-status]{detail}")

    def build_env(self, bind_core_offset=None):
        env = os.environ.copy()
        if self.validate_routing:
            env["RECSTORE_RDMA_VALIDATE_ROUTING"] = "1"
        if bind_core_offset is not None:
            env["RECSTORE_BIND_CORE_OFFSET"] = str(bind_core_offset)
        return env

    def server_bind_core_stride(self):
        get_workers = self.rdma_server_get_workers or 0
        return max(1, self.thread_num + get_workers)

    def build_server_env(self, global_id=0):
        offset = self.rdma_server_bind_core_offset
        if offset is not None:
            offset += global_id * self.server_bind_core_stride()
        return self.build_env(bind_core_offset=offset)

    def client_bind_core_offset(self, client_index):
        offset = self.rdma_client_bind_core_offset
        if offset is None:
            return None
        stride = self.rdma_client_bind_core_stride
        if stride is None:
            stride = 1
        return offset + client_index * stride

    def build_client_env(self, client_index=0):
        return self.build_env(
            bind_core_offset=self.client_bind_core_offset(client_index)
        )

    def build_server_cmd(self, global_id):
        cmd = [
            str(self.server_path),
            f"--config_path={self.config_path}",
            f"--global_id={global_id}",
            f"--num_server_processes={self.num_servers}",
            f"--num_client_processes={self.num_clients}",
            f"--rdma_rc_num_logical_clients={self.logical_num_clients}",
            f"--thread_num={self.thread_num}",
            f"--value_size={self.value_size}",
            f"--max_kv_num_per_request={self.max_kv_num_per_request}",
            "--use_dram=true",
            f"--rdma_rc_namespace={self.rdma_namespace}",
            f"--rdma_control_plane_host={self.rdma_control_plane_host}",
            f"--rdma_control_plane_port={self.rdma_control_plane_port}",
        ]
        if self.rdma_per_thread_response_limit_bytes is not None:
            cmd.append(
                "--rdma_per_thread_response_limit_bytes="
                f"{self.rdma_per_thread_response_limit_bytes}"
            )
        if self.rdma_put_server_scratch_bytes is not None:
            cmd.append(
                "--rdma_put_server_scratch_bytes="
                f"{self.rdma_put_server_scratch_bytes}"
            )
        if self.rdma_put_v2_push_slot_bytes is not None:
            cmd.append(
                "--rdma_put_v2_push_slot_bytes="
                f"{self.rdma_put_v2_push_slot_bytes}"
            )
        if self.rdma_put_v2_push_slots_per_client is not None:
            cmd.append(
                "--rdma_put_v2_push_slots_per_client="
                f"{self.rdma_put_v2_push_slots_per_client}"
            )
        if self.rdma_put_v2_push_region_offset is not None:
            cmd.append(
                "--rdma_put_v2_push_region_offset="
                f"{self.rdma_put_v2_push_region_offset}"
            )
        if self.rdma_qps_per_client_per_shard is not None:
            cmd.append(
                "--rdma_rc_qps_per_client_per_shard="
                f"{self.rdma_qps_per_client_per_shard}"
            )
        if self.rdma_slots_per_qp is not None:
            cmd.append(f"--rdma_rc_slots_per_qp={self.rdma_slots_per_qp}")
        if self.rdma_profile_interval_ms is not None:
            cmd.append(
                "--rdma_rc_profile_interval_ms="
                f"{self.rdma_profile_interval_ms}"
            )
        if self.rdma_server_coroutines_per_thread is not None:
            cmd.append(
                "--rdma_rc_server_coroutines_per_thread="
                f"{self.rdma_server_coroutines_per_thread}"
            )
        if self.rdma_server_get_workers is not None:
            cmd.append(
                "--rdma_rc_server_get_workers="
                f"{self.rdma_server_get_workers}"
            )
        if self.rdma_wait_timeout_ms is not None:
            cmd.append(f"--rdma_wait_timeout_ms={self.rdma_wait_timeout_ms}")
        if self.rdma_control_plane_timeout_ms is not None:
            cmd.append(
                "--rdma_control_plane_timeout_ms="
                f"{self.rdma_control_plane_timeout_ms}"
            )
        if self.rdma_inline_bytes is not None:
            cmd.append(f"--rdma_rc_inline_bytes={self.rdma_inline_bytes}")
        if self.rdma_server_numa_id is not None:
            cmd.append(f"--rdma_rc_server_numa_id={self.rdma_server_numa_id}")
            cmd.append(f"--numa_id={self.rdma_server_numa_id}")
        if self.rdma_fake_get_mode is not None:
            cmd.append(f"--rdma_rc_fake_get_mode={self.rdma_fake_get_mode}")
        if self.server_command_wrapper is not None:
            cmd = self.server_command_wrapper(global_id, cmd)
        return cmd

    def build_client_cmd(self, argv, client_index=0):
        client_global_id = self.num_servers + client_index
        client_id_base = client_index * self.logical_clients_per_process
        cmd = list(argv) + [
            f"--global_id={client_global_id}",
            f"--num_server_processes={self.num_servers}",
            f"--num_client_processes={self.num_clients}",
            f"--rdma_rc_num_logical_clients={self.logical_num_clients}",
            f"--rdma_rc_client_id_base={client_id_base}",
            f"--value_size={self.value_size}",
            f"--max_kv_num_per_request={self.max_kv_num_per_request}",
            f"--rdma_rc_namespace={self.rdma_namespace}",
            f"--rdma_control_plane_host={self.rdma_control_plane_host}",
            f"--rdma_control_plane_port={self.rdma_control_plane_port}",
        ]
        if self.rdma_client_receive_arena_bytes is not None:
            cmd.append(
                "--rdma_client_receive_arena_bytes="
                f"{self.rdma_client_receive_arena_bytes}"
            )
        if _supports_put_v2_client_flags(argv):
            if self.rdma_put_protocol_version is not None:
                cmd.append(
                    "--rdma_put_protocol_version="
                    f"{self.rdma_put_protocol_version}"
                )
            if self.rdma_put_v2_transfer_mode is not None:
                cmd.append(
                    "--rdma_put_v2_transfer_mode="
                    f"{self.rdma_put_v2_transfer_mode}"
                )
            if self.rdma_put_v2_push_slot_bytes is not None:
                cmd.append(
                    "--rdma_put_v2_push_slot_bytes="
                    f"{self.rdma_put_v2_push_slot_bytes}"
                )
            if self.rdma_put_v2_push_slots_per_client is not None:
                cmd.append(
                    "--rdma_put_v2_push_slots_per_client="
                    f"{self.rdma_put_v2_push_slots_per_client}"
                )
            if self.rdma_put_v2_push_region_offset is not None:
                cmd.append(
                    "--rdma_put_v2_push_region_offset="
                    f"{self.rdma_put_v2_push_region_offset}"
                )
            if self.rdma_put_client_send_arena_bytes is not None:
                cmd.append(
                    "--rdma_put_client_send_arena_bytes="
                    f"{self.rdma_put_client_send_arena_bytes}"
                )
        if self.rdma_qps_per_client_per_shard is not None:
            cmd.append(
                "--rdma_rc_qps_per_client_per_shard="
                f"{self.rdma_qps_per_client_per_shard}"
            )
        if self.rdma_slots_per_qp is not None:
            cmd.append(f"--rdma_rc_slots_per_qp={self.rdma_slots_per_qp}")
        if self.rdma_wait_timeout_ms is not None:
            cmd.append(f"--rdma_wait_timeout_ms={self.rdma_wait_timeout_ms}")
        if self.rdma_control_plane_timeout_ms is not None:
            cmd.append(
                "--rdma_control_plane_timeout_ms="
                f"{self.rdma_control_plane_timeout_ms}"
            )
        if self.rdma_profile_interval_ms is not None:
            cmd.append(
                "--rdma_rc_profile_interval_ms="
                f"{self.rdma_profile_interval_ms}"
            )
        if self.rdma_inline_bytes is not None:
            cmd.append(f"--rdma_rc_inline_bytes={self.rdma_inline_bytes}")
        client_numa_id = self.rdma_client_numa_id
        if self.rdma_client_numa_ids is not None:
            client_numa_id = self.rdma_client_numa_ids[client_index]
        if client_numa_id is not None:
            cmd.append(f"--rdma_rc_client_numa_id={client_numa_id}")
        if self.rdma_skip_client_copy is not None:
            cmd.append(
                "--rdma_rc_skip_client_copy="
                f"{str(self.rdma_skip_client_copy).lower()}"
            )
        return cmd

    def is_ready_line(self, line):
        return (
            "[RDMA-DBG] Server polling thread ready" in line
            or "component=rdma_server event=polling_thread_ready" in line
        )

    def _extract_ready_thread_token(self, line):
        if "component=rdma_server event=polling_thread_ready" in line:
            match = re.search(r"thread_id=(\d+)", line)
            if match is not None:
                return match.group(1)
        return line.rsplit(" ", 1)[-1]

    def _monitor(self, global_id, pipe):
        for raw_line in iter(pipe.readline, ""):
            line = raw_line.rstrip()
            self.process_logs.setdefault(global_id, []).append(line)
            if self.verbose:
                print(f"[petps_server:{global_id}] {line}")
            if self.is_ready_line(line):
                ready = self.ready_threads.setdefault(global_id, set())
                ready.add(self._extract_ready_thread_token(line))
                if len(ready) >= self.thread_num:
                    self.ready.add(global_id)

    def _format_captured_process_output(self, global_id):
        lines = self.process_logs.get(global_id, [])
        if not lines:
            return ""
        return (
            f"\nCaptured output from petps_server[{global_id}] "
            f"(last {len(lines)} lines):\n" + "\n".join(lines)
        )

    def _start_server_process(self, global_id, env):
        process = subprocess.Popen(
            self.build_server_cmd(global_id),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
            env=env,
        )
        thread = threading.Thread(
            target=self._monitor, args=(global_id, process.stdout), daemon=True
        )
        thread.start()
        self.processes.append((process, thread))
        return process

    def _wait_for_control_plane_ready(self, shard0_process):
        deadline = time.time() + self.timeout
        next_refresh = time.time() + self.status_refresh_interval
        while True:
            if shard0_process.poll() is not None:
                crash_details = self._format_captured_process_output(0)
                self.stop()
                raise RuntimeError(
                    "shard-0 petps_server exited early with code "
                    f"{shard0_process.returncode}{crash_details}"
                )
            try:
                with socket.create_connection(
                    (self.rdma_control_plane_host, self.rdma_control_plane_port),
                    timeout=0.5,
                ):
                    if self.show_control_plane_logs:
                        print(
                            "[petps-control-plane] ready "
                            f"host={self.rdma_control_plane_host} "
                            f"port={self.rdma_control_plane_port}"
                        )
                    return
            except OSError:
                pass

            if time.time() > deadline:
                self.stop()
                raise TimeoutError(
                    "Timed out waiting for shard-0 control plane at "
                    f"{self.rdma_control_plane_host}:{self.rdma_control_plane_port}"
                )
            if (
                self.status_refresh_interval > 0
                and time.time() >= next_refresh
            ):
                self.emit_status(
                    "control-plane-wait",
                    f"host={self.rdma_control_plane_host}:{self.rdma_control_plane_port}",
                )
                next_refresh = time.time() + self.status_refresh_interval
            time.sleep(0.2)

    def start(self):
        if self.server_command_wrapper is None and not self.server_path.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_path}")

        env = self.build_server_env(0)
        if self.show_control_plane_logs:
            print(
                "[petps-control-plane] config "
                f"host={self.rdma_control_plane_host} "
                f"port={self.rdma_control_plane_port} "
                f"namespace={self.rdma_namespace}"
            )

        shard0 = self._start_server_process(0, env)
        self._wait_for_control_plane_ready(shard0)

        for global_id in range(1, self.num_servers):
            self._start_server_process(global_id, self.build_server_env(global_id))

        if self.startup_delay > 0:
            time.sleep(self.startup_delay)

        if not self.ready:
            for global_id, (process, _thread) in enumerate(self.processes):
                if process.poll() is None:
                    self.ready.add(global_id)

        deadline = time.time() + self.timeout
        next_refresh = time.time() + self.status_refresh_interval
        while len(self.ready) < self.num_servers:
            if time.time() > deadline:
                self.emit_status("startup-timeout", f"timeout={self.timeout}s")
                self.stop()
                raise TimeoutError(
                    f"Timed out waiting for {self.num_servers} petps_server processes"
                )
            for idx, (process, _thread) in enumerate(self.processes):
                if process.poll() is not None:
                    self.emit_status("startup-crash", f"exit_code={process.returncode}")
                    crash_details = self._format_captured_process_output(idx)
                    self.stop()
                    raise RuntimeError(
                        "petps_server exited early with code "
                        f"{process.returncode}{crash_details}"
                    )
            if (
                self.status_refresh_interval > 0
                and time.time() >= next_refresh
            ):
                self.emit_status("startup-wait")
                next_refresh = time.time() + self.status_refresh_interval
            time.sleep(0.2)

    def run_client(self, argv, client_index=0, stream_output=True, timeout=None):
        cmd = self.build_client_cmd(argv, client_index=client_index)
        env = self.build_client_env(client_index)
        if self.show_control_plane_logs:
            print(
                "[petps-control-plane] client "
                f"host={self.rdma_control_plane_host} "
                f"port={self.rdma_control_plane_port} "
                f"client_index={client_index}"
            )
        if not stream_output:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    text=True,
                    capture_output=True,
                    check=False,
                    env=env,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                class Completed:
                    def __init__(self, stdout, stderr):
                        self.returncode = 124
                        self.stdout = stdout
                        self.stderr = stderr

                timeout_text = (
                    f"\n[petps-client] timed out after {timeout} seconds\n"
                )
                stdout = _to_text(exc.stdout) + timeout_text
                stderr = _to_text(exc.stderr)
                return Completed(stdout, stderr)
            if self.verbose:
                print(completed.stdout)
                print(completed.stderr)
            return completed

        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )

        output_lines = []
        line_queue = queue.Queue()

        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        deadline = time.monotonic() + timeout if timeout is not None else None
        returncode = None
        timed_out = False
        reader_done = False

        while True:
            try:
                line = line_queue.get(timeout=0.05)
                if line is None:
                    reader_done = True
                else:
                    output_lines.append(line)
                    print(line, end="")
            except queue.Empty:
                pass

            if returncode is None:
                returncode = process.poll()

            if (
                deadline is not None
                and time.monotonic() >= deadline
                and returncode is None
            ):
                timed_out = True
                break

            if returncode is not None and reader_done:
                break

        if timed_out:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            timeout_line = f"[petps-client] timed out after {timeout} seconds\n"
            output_lines.append(timeout_line)
            print(timeout_line, end="")
            returncode = 124
        elif returncode is None:
            returncode = process.wait()

        reader.join(timeout=1)
        while True:
            try:
                line = line_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            output_lines.append(line)
            print(line, end="")

        class Completed:
            def __init__(self, returncode, stdout):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = ""

        return Completed(returncode, "".join(output_lines))

    def stop(self):
        for process, thread in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            thread.join(timeout=1)
        self.processes.clear()
        self.process_logs.clear()
        self.ready.clear()
        self.ready_threads.clear()

    @contextmanager
    def run(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()
