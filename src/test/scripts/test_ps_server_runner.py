import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ps_server_runner import PSServerRunner


class TestPSServerRunner(unittest.TestCase):
    def test_start_returns_true_on_ready_line(self):
        proc = mock.Mock()
        proc.stdout = iter(["[STDOUT] listening\n", "READY\n"])
        proc.poll.return_value = None

        with mock.patch("ps_server_runner.subprocess.Popen", return_value=proc):
            runner = PSServerRunner(
                server_path=__file__,
                launcher_cli="/tmp/ps_server_launcher_cli",
                verbose=False,
            )
            self.assertTrue(runner.start())
            self.assertIs(proc, runner.process)

    def test_start_returns_true_on_skip_line(self):
        proc = mock.Mock()
        proc.stdout = iter(["SKIP\talready_running\n"])
        proc.poll.return_value = 0

        with mock.patch("ps_server_runner.subprocess.Popen", return_value=proc):
            runner = PSServerRunner(
                server_path=__file__,
                launcher_cli="/tmp/ps_server_launcher_cli",
            )
            self.assertTrue(runner.start())
            self.assertIsNone(runner.process)
            self.assertEqual(runner._skip_reason, "already_running")

    def test_serve_command_forwards_launcher_options(self):
        runner = PSServerRunner(
            server_path=__file__,
            config_path=__file__,
            log_dir="/tmp/recstore_ps_test",
            timeout=17,
            num_shards=3,
            startup_delay=1.5,
            launcher_cli="/tmp/ps_server_launcher_cli",
            verbose=True,
        )

        self.assertEqual(
            runner._serve_command(),
            [
                "/tmp/ps_server_launcher_cli",
                "serve",
                "--server-path",
                __file__,
                "--log-dir",
                "/tmp/recstore_ps_test",
                "--timeout",
                "17",
                "--num-shards",
                "3",
                "--startup-delay-ms",
                "1500",
                "--config",
                __file__,
                "--verbose",
            ],
        )


if __name__ == "__main__":
    unittest.main()
