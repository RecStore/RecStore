from __future__ import annotations

import unittest
from unittest import mock

from model_zoo.rs_demo import config
from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners import recstore_runner


class EmitStepMetricsTest(unittest.TestCase):
    def test_emits_curated_metrics_only(self) -> None:
        row = {
            "step_total_ms": 12.5,
            "samples_per_sec": 8000.0,
            "emb_stage_ms": 3.0,
            "loss": 0.5,
            "warmup_excluded": 1,
            "embed_lookup_ms": 2.0,
            "unrelated_column": 99.0,
        }
        with mock.patch("recstore.telemetry.emit") as emit:
            recstore_runner._emit_step_metrics(row, "run-1", 3)

        self.assertEqual(
            [call.args[2] for call in emit.call_args_list],
            [
                "step_total_ms",
                "samples_per_sec",
                "emb_stage_ms",
                "embed_lookup_ms",
                "loss",
                "warmup_excluded",
            ],
        )
        for call in emit.call_args_list:
            self.assertEqual(call.args[0], "e2e_step")
            self.assertEqual(call.args[1], "run-1|step3")
        self.assertEqual(emit.call_args_list[0].args[3], 12.5)

    def test_skips_blank_and_non_numeric_values(self) -> None:
        row = {
            "loss": "",
            "step_total_ms": None,
            "emb_stage_ms": "n/a",
            "samples_per_sec": "123",
        }
        with mock.patch("recstore.telemetry.emit") as emit:
            recstore_runner._emit_step_metrics(row, "run-2", 0)

        self.assertEqual(
            [call.args[2] for call in emit.call_args_list], ["samples_per_sec"]
        )
        self.assertEqual(emit.call_args_list[0].args[3], 123.0)

    def test_restricted_names_emit_only_that_metric(self) -> None:
        row = {"loss": 0.375, "step_total_ms": 12.5, "samples_per_sec": 8000.0}
        with mock.patch("recstore.telemetry.emit") as emit:
            recstore_runner._emit_step_metrics(
                row, "run-3", 7, names=("loss",)
            )

        self.assertEqual([call.args[2] for call in emit.call_args_list], ["loss"])
        self.assertEqual(emit.call_args_list[0].args[1], "run-3|step7")
        self.assertEqual(emit.call_args_list[0].args[3], 0.375)


class ReportMetricsConfigTest(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        self.assertFalse(RunConfig().report_metrics)

    def test_cli_flag_enables_it(self) -> None:
        parser = config.build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.report_metrics)
        args = parser.parse_args(["--report-metrics"])
        self.assertTrue(args.report_metrics)


if __name__ == "__main__":
    unittest.main()
