import unittest
from pathlib import Path
from unittest import mock

from model_zoo.rs_demo.runtime.report import analyze_embupdate


class ReportRuntimeTest(unittest.TestCase):
    def test_analyze_embupdate_includes_server_log_when_present(self) -> None:
        repo_root = Path("/repo")
        with mock.patch("model_zoo.rs_demo.runtime.report.subprocess.run") as run_mock:
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
                "--export-csv",
                "/tmp/recstore_embupdate.csv",
                "--top",
                "7",
            ],
        )


if __name__ == "__main__":
    unittest.main()
