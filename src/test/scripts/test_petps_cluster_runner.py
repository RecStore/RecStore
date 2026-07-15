import socket
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from petps_cluster_runner import PetPSClusterRunner


class TestPetPSClusterRunner(unittest.TestCase):
    def test_build_server_command_includes_control_plane_flags(self):
        runner = PetPSClusterRunner(
            server_path="./build/bin/petps_server",
            config_path="./recstore_config.json",
            num_servers=2,
            num_clients=1,
            thread_num=2,
            value_size=16,
            max_kv_num_per_request=64,
            rdma_namespace="bench-ns",
            rdma_control_plane_host="127.0.0.2",
            rdma_control_plane_port=32000,
        )

        cmd = runner.build_server_cmd(1)

        self.assertIn("--global_id=1", cmd)
        self.assertIn("--num_server_processes=2", cmd)
        self.assertIn("--num_client_processes=1", cmd)
        self.assertIn("--rdma_rc_num_logical_clients=1", cmd)
        self.assertIn("--thread_num=2", cmd)
        self.assertIn("--value_size=16", cmd)
        self.assertIn("--max_kv_num_per_request=64", cmd)
        self.assertIn("--rdma_rc_namespace=bench-ns", cmd)
        self.assertIn("--rdma_control_plane_host=127.0.0.2", cmd)
        self.assertIn("--rdma_control_plane_port=32000", cmd)

    def test_build_server_command_uses_optional_wrapper(self):
        runner = PetPSClusterRunner(
            server_path="/tmp/petps_server",
            config_path="/tmp/recstore_config.json",
            rdma_control_plane_port=32000,
            server_command_wrapper=lambda global_id, cmd: [
                "ssh",
                f"host{global_id}",
                " ".join(cmd),
            ],
        )

        cmd = runner.build_server_cmd(1)

        self.assertEqual(cmd[0:2], ["ssh", "host1"])
        self.assertIn("/tmp/petps_server", cmd[2])
        self.assertIn("--global_id=1", cmd[2])

    def test_build_client_command_assigns_client_global_id(self):
        runner = PetPSClusterRunner(
            num_servers=2,
            num_clients=1,
            rdma_namespace="bench-ns",
            rdma_control_plane_port=25001,
        )
        cmd = runner.build_client_cmd(
            ["./build/bin/petps_integration_test"],
            client_index=0,
        )
        self.assertIn("--global_id=2", cmd)
        self.assertIn("--num_server_processes=2", cmd)
        self.assertIn("--num_client_processes=1", cmd)
        self.assertIn("--rdma_rc_client_id_base=0", cmd)
        self.assertIn("--rdma_rc_namespace=bench-ns", cmd)
        self.assertIn("--rdma_control_plane_port=25001", cmd)

    def test_build_commands_use_logical_client_count_for_multi_thread_clients(self):
        runner = PetPSClusterRunner(
            num_servers=1,
            num_clients=3,
            logical_clients_per_process=4,
            rdma_control_plane_port=25005,
        )

        server_cmd = runner.build_server_cmd(0)
        client2_cmd = runner.build_client_cmd(
            ["./build/bin/ps_transport_benchmark"],
            client_index=2,
        )

        self.assertIn("--num_client_processes=3", server_cmd)
        self.assertIn("--rdma_rc_num_logical_clients=12", server_cmd)
        self.assertIn("--global_id=3", client2_cmd)
        self.assertIn("--num_client_processes=3", client2_cmd)
        self.assertIn("--rdma_rc_num_logical_clients=12", client2_cmd)
        self.assertIn("--rdma_rc_client_id_base=8", client2_cmd)

    def test_build_env_includes_validate_routing_only(self):
        runner = PetPSClusterRunner(validate_routing=True)
        env = runner.build_env()
        self.assertEqual(env["RECSTORE_RDMA_VALIDATE_ROUTING"], "1")
        self.assertNotIn("RECSTORE_MEMCACHED_HOST", env)

    def test_build_env_supports_bind_core_offset(self):
        runner = PetPSClusterRunner(
            rdma_server_bind_core_offset=4,
            thread_num=2,
            rdma_server_get_workers=1,
            rdma_client_bind_core_offset=20,
            rdma_client_bind_core_stride=3,
        )

        server_env = runner.build_server_env(0)
        server1_env = runner.build_server_env(1)
        client0_env = runner.build_client_env(0)
        client2_env = runner.build_client_env(2)

        self.assertEqual(server_env["RECSTORE_BIND_CORE_OFFSET"], "4")
        self.assertEqual(server1_env["RECSTORE_BIND_CORE_OFFSET"], "7")
        self.assertEqual(client0_env["RECSTORE_BIND_CORE_OFFSET"], "20")
        self.assertEqual(client2_env["RECSTORE_BIND_CORE_OFFSET"], "26")

    def test_auto_namespace_uses_generated_value(self):
        runner = PetPSClusterRunner(rdma_namespace="auto")
        self.assertTrue(runner.rdma_namespace.startswith("recstore-rdma-"))

    def test_allocates_control_plane_port_when_unspecified(self):
        runner = PetPSClusterRunner(
            rdma_control_plane_host="127.0.0.1",
            rdma_control_plane_port=None,
        )
        self.assertGreater(runner.rdma_control_plane_port, 0)

    def test_detects_ready_lines(self):
        runner = PetPSClusterRunner()
        self.assertTrue(
            runner.is_ready_line("[RDMA-DBG] Server polling thread ready 0")
        )
        self.assertTrue(
            runner.is_ready_line(
                "component=rdma_server event=polling_thread_ready thread_id=0"
            )
        )
        self.assertFalse(runner.is_ready_line("Starts PS polling thread 0"))

    def test_monitor_requires_all_polling_threads_before_marking_ready(self):
        runner = PetPSClusterRunner(num_servers=1, thread_num=2)
        pipe = StringIO(
            "component=rdma_server event=polling_thread_ready thread_id=0\n"
            "component=rdma_server event=polling_thread_ready thread_id=1\n"
        )

        runner._monitor(0, pipe)

        self.assertEqual(runner.ready, {0})

    def test_build_commands_include_optional_rdma_flags(self):
        runner = PetPSClusterRunner(
            rdma_namespace="bench-ns",
            rdma_control_plane_host="127.0.0.1",
            rdma_control_plane_port=25002,
            rdma_per_thread_response_limit_bytes=2097152,
            rdma_client_receive_arena_bytes=134217728,
            rdma_qps_per_client_per_shard=4,
            rdma_slots_per_qp=6,
            rdma_server_coroutines_per_thread=3,
            rdma_server_get_workers=2,
            rdma_inline_bytes=64,
            rdma_client_numa_id=1,
            rdma_server_numa_id=2,
        )
        server_cmd = runner.build_server_cmd(0)
        client_cmd = runner.build_client_cmd(["./build/bin/petps_integration_test"])
        self.assertIn("--rdma_per_thread_response_limit_bytes=2097152", server_cmd)
        self.assertIn("--rdma_rc_qps_per_client_per_shard=4", server_cmd)
        self.assertIn("--rdma_rc_slots_per_qp=6", server_cmd)
        self.assertIn("--rdma_rc_server_coroutines_per_thread=3", server_cmd)
        self.assertIn("--rdma_rc_server_get_workers=2", server_cmd)
        self.assertIn("--rdma_rc_inline_bytes=64", server_cmd)
        self.assertIn("--rdma_rc_server_numa_id=2", server_cmd)
        self.assertIn("--rdma_client_receive_arena_bytes=134217728", client_cmd)
        self.assertIn("--rdma_rc_qps_per_client_per_shard=4", client_cmd)
        self.assertIn("--rdma_rc_slots_per_qp=6", client_cmd)
        self.assertIn("--rdma_rc_inline_bytes=64", client_cmd)
        self.assertIn("--rdma_rc_client_numa_id=1", client_cmd)

    def test_build_client_command_supports_per_client_numa_ids(self):
        runner = PetPSClusterRunner(
            num_servers=1,
            num_clients=2,
            rdma_client_numa_ids=[0, 1],
        )
        client0_cmd = runner.build_client_cmd(
            ["./build/bin/petps_integration_test"],
            client_index=0,
        )
        client1_cmd = runner.build_client_cmd(
            ["./build/bin/petps_integration_test"],
            client_index=1,
        )
        self.assertIn("--rdma_rc_client_numa_id=0", client0_cmd)
        self.assertIn("--rdma_rc_client_numa_id=1", client1_cmd)

    def test_build_client_command_allows_benchmark_binary_without_put_v2_flags(self):
        runner = PetPSClusterRunner(
            rdma_put_protocol_version=2,
            rdma_put_v2_transfer_mode="read",
            rdma_put_v2_push_slot_bytes=262144,
        )

        cmd = runner.build_client_cmd(
            ["./build/bin/ps_transport_benchmark"],
            client_index=0,
        )

        rendered = " ".join(cmd)
        self.assertNotIn("--rdma_put_protocol_version=", rendered)
        self.assertNotIn("--rdma_put_v2_transfer_mode=", rendered)
        self.assertNotIn("--rdma_put_v2_push_slot_bytes=", rendered)

    def test_build_commands_accept_legacy_rdma_aliases(self):
        runner = PetPSClusterRunner(
            rdma_rc_qps_per_client_per_shard=8,
            rdma_rc_slots_per_qp=5,
            rdma_rc_profile_interval_ms=250,
            rdma_rc_server_coroutines_per_thread=2,
            rdma_rc_server_get_workers=3,
            rdma_rc_inline_bytes=48,
            rdma_rc_client_numa_id=1,
            rdma_rc_server_numa_id=2,
            rdma_rc_fake_get_mode="status_only",
            rdma_rc_skip_client_copy=True,
        )
        server_cmd = runner.build_server_cmd(0)
        client_cmd = runner.build_client_cmd(["./build/bin/petps_integration_test"])
        self.assertEqual(runner.rdma_qps_per_client_per_shard, 8)
        self.assertEqual(runner.rdma_slots_per_qp, 5)
        self.assertEqual(runner.rdma_profile_interval_ms, 250)
        self.assertEqual(runner.rdma_server_coroutines_per_thread, 2)
        self.assertEqual(runner.rdma_server_get_workers, 3)
        self.assertEqual(runner.rdma_inline_bytes, 48)
        self.assertEqual(runner.rdma_client_numa_id, 1)
        self.assertEqual(runner.rdma_server_numa_id, 2)
        self.assertEqual(runner.rdma_fake_get_mode, "status_only")
        self.assertTrue(runner.rdma_skip_client_copy)
        self.assertIn("--rdma_rc_qps_per_client_per_shard=8", server_cmd)
        self.assertIn("--rdma_rc_slots_per_qp=5", server_cmd)
        self.assertIn("--rdma_rc_profile_interval_ms=250", server_cmd)
        self.assertIn("--rdma_rc_server_coroutines_per_thread=2", server_cmd)
        self.assertIn("--rdma_rc_server_get_workers=3", server_cmd)
        self.assertIn("--rdma_rc_inline_bytes=48", server_cmd)
        self.assertIn("--rdma_rc_server_numa_id=2", server_cmd)
        self.assertIn("--rdma_rc_fake_get_mode=status_only", server_cmd)
        self.assertIn("--rdma_rc_slots_per_qp=5", client_cmd)
        self.assertIn("--rdma_rc_inline_bytes=48", client_cmd)
        self.assertIn("--rdma_rc_client_numa_id=1", client_cmd)
        self.assertIn("--rdma_rc_skip_client_copy=true", client_cmd)

    @mock.patch("petps_cluster_runner.socket.create_connection")
    def test_wait_for_control_plane_ready_succeeds(self, mock_connect):
        runner = PetPSClusterRunner(
            timeout=1,
            status_refresh_interval=0,
            rdma_control_plane_port=25003,
        )
        process = mock.Mock()
        process.poll.return_value = None
        mock_connect.return_value.__enter__.return_value = mock.Mock()

        runner._wait_for_control_plane_ready(process)

        mock_connect.assert_called_once_with(("127.0.0.1", 25003), timeout=0.5)

    @mock.patch("petps_cluster_runner.socket.create_connection")
    def test_wait_for_control_plane_ready_reports_crash_logs(self, mock_connect):
        runner = PetPSClusterRunner(
            timeout=1,
            status_refresh_interval=0,
            rdma_control_plane_port=25004,
        )
        process = mock.Mock()
        process.poll.return_value = -11
        process.returncode = -11
        runner.process_logs[0] = ["ib device wasn't found"]

        with mock.patch.object(runner, "stop"), self.assertRaises(RuntimeError) as ctx:
            runner._wait_for_control_plane_ready(process)

        self.assertIn("shard-0 petps_server exited early with code -11", str(ctx.exception))
        self.assertIn("ib device wasn't found", str(ctx.exception))
        mock_connect.assert_not_called()

    def test_start_launches_shard0_before_other_servers(self):
        runner = PetPSClusterRunner(
            num_servers=3,
            startup_delay=0,
            timeout=0,
            status_refresh_interval=0,
        )
        runner.server_path = mock.Mock()
        runner.server_path.exists.return_value = True

        shard0_proc = mock.Mock()
        shard0_proc.poll.return_value = None
        shard1_proc = mock.Mock()
        shard1_proc.poll.return_value = None
        shard2_proc = mock.Mock()
        shard2_proc.poll.return_value = None

        start_order = []

        def fake_start_server(global_id, _env):
            start_order.append(global_id)
            process = [shard0_proc, shard1_proc, shard2_proc][global_id]
            runner.processes.append((process, mock.Mock()))
            runner.ready.add(global_id)
            return process

        with mock.patch.object(runner, "_start_server_process", side_effect=fake_start_server), \
             mock.patch.object(runner, "_wait_for_control_plane_ready") as mock_wait:
            runner.start()

        self.assertEqual(start_order, [0, 1, 2])
        mock_wait.assert_called_once_with(shard0_proc)

    def test_emit_status_prints_ready_and_pid_info(self):
        runner = PetPSClusterRunner(num_servers=2)
        runner.ready.add(0)
        fake_proc = mock.Mock()
        fake_proc.pid = 1234
        fake_proc.poll.return_value = None
        runner.processes = [(fake_proc, mock.Mock())]

        with mock.patch("sys.stdout", new_callable=StringIO) as fake_out:
            runner.emit_status("startup-wait")

        output = fake_out.getvalue()
        self.assertIn("[petps-status]", output)
        self.assertIn("ready=1/2", output)
        self.assertIn("running_pids=1234", output)

    def test_startup_crash_error_includes_captured_server_output(self):
        runner = PetPSClusterRunner(num_servers=1)
        runner.server_path = mock.Mock()
        runner.server_path.exists.return_value = True
        runner.startup_delay = 0
        runner.timeout = 1
        runner.status_refresh_interval = 0

        fake_proc = mock.Mock()
        fake_proc.poll.return_value = -11
        fake_proc.returncode = -11
        runner.process_logs[0] = [
            "set NUMA ID = 0",
            "ib device wasn't found",
        ]

        def fake_start_server(_global_id, _env):
            runner.processes.append((fake_proc, mock.Mock()))
            return fake_proc

        with mock.patch.object(runner, "_start_server_process", side_effect=fake_start_server), \
             mock.patch.object(runner, "_wait_for_control_plane_ready"), \
             mock.patch.object(runner, "stop"), \
             self.assertRaises(RuntimeError) as ctx:
            runner.start()

        message = str(ctx.exception)
        self.assertIn("petps_server exited early with code -11", message)
        self.assertIn("Captured output from petps_server[0]", message)
        self.assertIn("ib device wasn't found", message)

    @mock.patch("petps_cluster_runner.subprocess.run")
    def test_run_client_timeout_handles_bytes_stdout_stderr(self, mock_run):
        runner = PetPSClusterRunner()
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["/bin/echo", "x"],
            timeout=1,
            output=b"partial-out",
            stderr=b"partial-err",
        )

        completed = runner.run_client(
            ["/bin/echo", "x"],
            stream_output=False,
            timeout=1,
        )

        self.assertEqual(completed.returncode, 124)
        self.assertIn("partial-out", completed.stdout)
        self.assertIn("timed out after 1 seconds", completed.stdout)
        self.assertEqual(completed.stderr, "partial-err")

    @mock.patch("petps_cluster_runner.subprocess.Popen")
    def test_run_client_stream_output_timeout_does_not_block_on_readline(
        self, mock_popen
    ):
        runner = PetPSClusterRunner()
        fake_process = mock.Mock()
        fake_process.stdout.readline.side_effect = ["partial\n", ""]
        fake_process.poll.return_value = None
        fake_process.wait.return_value = 0
        fake_process.pid = 1234
        mock_popen.return_value = fake_process

        completed = runner.run_client(
            ["/bin/echo", "x"],
            stream_output=True,
            timeout=1,
        )

        self.assertEqual(completed.returncode, 124)
        self.assertIn("partial", completed.stdout)
        self.assertIn("timed out after 1 seconds", completed.stdout)
        fake_process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
