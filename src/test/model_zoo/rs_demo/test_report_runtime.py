import os
import unittest
from pathlib import Path
from unittest import mock

from python.pytorch.recstore.benchmark.report import (
    analyze_embupdate,
    analyze_stage_table,
    setup_local_report_env,
)


class ReportRuntimeTest(unittest.TestCase):
    def test_setup_local_report_env_uses_jsonl_only_by_default(self) -> None:
        with mock.patch("python.pytorch.recstore.benchmark.report.Path.mkdir"), mock.patch(
            "builtins.open", mock.mock_open()
        ), mock.patch.dict("python.pytorch.recstore.benchmark.report.os.environ", {}, clear=True):
            setup_local_report_env("/tmp/recstore_events.jsonl")

            self.assertEqual("jsonl", os.environ["RECSTORE_REPORT_LOCAL_SINK"])
            self.assertEqual("256", os.environ["RECSTORE_REPORT_FLUSH_EVERY_N"])
            self.assertEqual("1", os.environ["RECSTORE_LOCAL_SHM_STAGE_REPORT"])

    def test_analyze_embupdate_includes_server_log_when_present(self) -> None:
        repo_root = Path("/repo")
        with mock.patch("python.pytorch.recstore.benchmark.report.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(
                returncode=0,
                stdout="ok",
                stderr="",
            )

            result = analyze_embupdate(
                repo_root=repo_root,
                jsonl_path="/tmp/recstore_events.jsonl",
                csv_path="/tmp/recstore_embupdate.csv",
                top_n=7,
                extra_inputs=["/tmp/ps_server.log"],
            )

        self.assertEqual(result, "ok")
        cmd = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args.args[0]
        self.assertEqual(
            cmd,
            [
                mock.ANY,
                "/repo/src/test/scripts/analyze_embupdate_stages.py",
                "--input",
                "/tmp/recstore_events.jsonl",
                "--input",
                "/tmp/ps_server.log",
                "--group-by-prefix",
                "--table-name",
                "embupdate_stages",
                "--export-csv",
                "/tmp/recstore_embupdate.csv",
                "--top",
                "7",
            ],
        )

    def test_analyze_stage_table_passes_requested_table_name(self) -> None:
        repo_root = Path("/repo")
        with mock.patch("python.pytorch.recstore.benchmark.report.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")

            result = analyze_stage_table(
                repo_root=repo_root,
                jsonl_path="/tmp/recstore_events.jsonl",
                csv_path="/tmp/local_shm.csv",
                table_name="local_shm_server_stages",
                top_n=3,
            )

        self.assertEqual(result, "ok")
        cmd = (
            run_mock.call_args.kwargs["args"]
            if "args" in run_mock.call_args.kwargs
            else run_mock.call_args.args[0]
        )
        self.assertIn("--table-name", cmd)
        self.assertEqual("local_shm_server_stages", cmd[cmd.index("--table-name") + 1])


if __name__ == "__main__":
    unittest.main()
