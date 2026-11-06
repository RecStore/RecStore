#!/usr/bin/env python3
"""Unit tests for the perf suite driver logic (no benchmark binaries needed).

These cover the parts that silently rot: metric parsing, median/spread
aggregation, baseline evaluation and the regression threshold semantics.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmarks import perf_suite

KV_OUTPUT = """
YCSB_LOAD_RESULT records=20000 threads=4 seconds=0.006 ops=20000 throughput_ops_sec=3300000.0
YCSB_RESULT workload=c distribution=uniform read_mode=batch_get_flat threads=4 records=20000 runtime_s=1.0 ops=1980 throughput_ops_sec=198237440.7 read_ops=1980 update_ops=0
YCSB_LATENCY samples=3098209 p50_us=1.299 p90_us=1.399 p99_us=1.399 max_us=31.900
"""

PS_OUTPUT = """
PS_BENCHMARK_RESULT phase=load transport=LOCAL_SHM mode=fetch threads=1 batch_size=8 records=2000 runtime_s=0.007 batches=250 key_ops=2000 throughput_batches_sec=35686.5 throughput_keys_sec=285492
PS_LOCAL_SHM_PROFILE phase=load samples=250 request_total_us_mean=23.6
PS_BENCHMARK_RESULT phase=run transport=LOCAL_SHM mode=fetch threads=1 batch_size=8 records=2000 runtime_s=1.0 batches=57556 key_ops=460448 throughput_batches_sec=57545.3 throughput_keys_sec=460362
PS_LOCAL_SHM_PROFILE phase=run samples=57556 request_total_us_mean=16.3278
"""


def metric(key: str, direction: str = "higher") -> perf_suite.MetricSpec:
    return perf_suite.MetricSpec(key=key, unit="unit", direction=direction)


class ParsingTest(unittest.TestCase):
    def test_kvengine_parses_run_metrics_not_load(self) -> None:
        metrics = perf_suite._parse_kvengine(KV_OUTPUT)
        self.assertAlmostEqual(metrics["throughput_ops_sec"], 198237440.7)
        self.assertAlmostEqual(metrics["p50_us"], 1.299)
        self.assertAlmostEqual(metrics["p99_us"], 1.399)

    def test_kvengine_requires_result_line(self) -> None:
        with self.assertRaises(perf_suite.SuiteError):
            perf_suite._parse_kvengine("nothing here")

    def test_kvengine_parses_tiered_cache_counters(self) -> None:
        output = (
            "YCSB_LOAD_RESULT records=10 threads=1 seconds=0.1 ops=10 "
            "throughput_ops_sec=100.0 tiered_cache_hit_rate=0.99\n"
            "YCSB_RESULT workload=c distribution=zipfian read_mode=batch_get_flat "
            "threads=16 records=10 runtime_s=1.0 ops=10 throughput_ops_sec=504800.0 "
            "read_ops=10 update_ops=0 tiered_cache_policy=lru tiered_cache_bytes=1024 "
            "tiered_cache_hits=90 tiered_cache_misses=10 tiered_cache_hit_rate=0.9\n"
        )
        metrics = perf_suite._parse_kvengine(output)
        self.assertAlmostEqual(metrics["tiered_cache_hit_rate"], 0.9)
        self.assertAlmostEqual(metrics["tiered_cache_hits"], 90.0)
        self.assertAlmostEqual(metrics["tiered_cache_bytes"], 1024.0)
        # Counters are optional: a `none` policy prints none of them.
        self.assertNotIn("tiered_cache_hit_rate", perf_suite._parse_kvengine(KV_OUTPUT))

    def test_ps_summary_uses_measure_phase(self) -> None:
        value = perf_suite._parse_float(
            perf_suite._PS_RESULT, PS_OUTPUT, "PS_BENCHMARK_RESULT"
        )
        self.assertAlmostEqual(value, 460362.0)
        profile = perf_suite._PS_PROFILE.search(PS_OUTPUT)
        assert profile is not None
        self.assertAlmostEqual(float(profile.group("value")), 16.3278)


class CommandTest(unittest.TestCase):
    @staticmethod
    def case(params: dict) -> perf_suite.Case:
        return perf_suite.Case(
            id="case-cmd",
            layer="storage",
            runner="kvengine",
            labels=("perf",),
            repeats=1,
            noise_floor_pct=5.0,
            params=params,
            metrics=(metric("throughput"),),
        )

    def test_kvengine_appends_optional_flags(self) -> None:
        work_dir = Path("/tmp/perf-work/r00")
        cmd = perf_suite._kvengine_command(
            self.case(
                {
                    "index_type": "DRAM_PET_HASH",
                    "value_store_type": "TIERED_VALUE_STORE",
                    "workload": "c",
                    "read_mode": "batch_get_flat",
                    "distribution": "zipfian",
                    "record_count": 1000,
                    "batch_keys": 64,
                    "thread_num": 16,
                    "running_seconds": 1,
                    "value_size": 128,
                    "zipfian_alpha": 0.9,
                    "zipfian_scramble": True,
                    "ssd_path": "{work_dir}/ssd",
                    "ssd_capacity_bytes": 268435456,
                    "ssd_io_backend": "IOURING",
                    "ssd_queue_depth": 512,
                }
            ),
            Path("/build"),
            work_dir,
        )
        self.assertIn(f"--ssd_path={work_dir}/ssd", cmd)
        self.assertIn("--zipfian_alpha=0.9", cmd)
        self.assertIn("--zipfian_scramble=true", cmd)
        self.assertIn("--ssd_capacity_bytes=268435456", cmd)
        self.assertIn("--ssd_io_backend=IOURING", cmd)
        self.assertIn("--ssd_queue_depth=512", cmd)

    def test_kvengine_omits_absent_optional_flags(self) -> None:
        cmd = perf_suite._kvengine_command(
            self.case(
                {
                    "index_type": "DRAM_PET_HASH",
                    "value_store_type": "DRAM_VALUE_STORE",
                    "workload": "c",
                    "read_mode": "batch_get_flat",
                    "distribution": "uniform",
                    "record_count": 1000,
                    "batch_keys": 64,
                    "thread_num": 4,
                    "running_seconds": 1,
                    "value_size": 128,
                }
            ),
            Path("/build"),
            Path("/tmp/perf-work/r00"),
        )
        for flag in ("ssd_path", "ssd_capacity_bytes", "ssd_io_backend",
                     "ssd_queue_depth", "zipfian_scramble", "zipfian_alpha",
                     "tiering_cache_policy", "tiering_cache_budget_ratio",
                     "tiering_cache_shards"):
            self.assertFalse(
                any(part.startswith(f"--{flag}") for part in cmd), flag
            )

    def test_kvengine_passes_tiering_flags(self) -> None:
        cmd = perf_suite._kvengine_command(
            self.case(
                {
                    "index_type": "DRAM_PET_HASH",
                    "value_store_type": "TIERED_VALUE_STORE",
                    "workload": "c",
                    "read_mode": "batch_get_flat",
                    "distribution": "zipfian",
                    "record_count": 1000,
                    "batch_keys": 64,
                    "thread_num": 16,
                    "running_seconds": 1,
                    "value_size": 128,
                    "tiering_cache_policy": "lru",
                    "tiering_cache_budget_ratio": 0.5,
                    "tiering_cache_shards": 16,
                }
            ),
            Path("/build"),
            Path("/tmp/perf-work/r00"),
        )
        self.assertIn("--tiering_cache_policy=lru", cmd)
        self.assertIn("--tiering_cache_budget_ratio=0.5", cmd)
        self.assertIn("--tiering_cache_shards=16", cmd)

    def test_clean_within_removes_stale_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "r00"
            stale = work_dir / "ssd"
            stale.mkdir(parents=True)
            (stale / "ssd.db").write_text("stale", encoding="utf-8")
            perf_suite._clean_within(work_dir, stale)
            self.assertFalse(stale.exists())

    def test_prepare_work_dir_clears_stale_ssd_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "r00"
            stale = work_dir / "ssd"
            stale.mkdir(parents=True)
            (stale / "ssd.db").write_text("stale", encoding="utf-8")
            perf_suite._prepare_work_dir(
                self.case({"ssd_path": "{work_dir}/ssd"}), work_dir
            )
            self.assertTrue(stale.is_dir())
            self.assertEqual(list(stale.iterdir()), [])
            self.assertTrue((work_dir / "dram").is_dir())

    def test_clean_within_refuses_outside_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "r00"
            work_dir.mkdir()
            with self.assertRaises(perf_suite.SuiteError):
                perf_suite._clean_within(work_dir, Path(tmp) / "elsewhere")


class AggregateTest(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict]:
        rows = []
        for repeat, value in enumerate([100.0, 102.0, 98.0]):
            for key, unit in (("throughput", "ops/s"), ("p99", "us")):
                rows.append(
                    {
                        "case": "case-a",
                        "metric": key,
                        "unit": unit,
                        "direction": "lower" if key == "p99" else "higher",
                        "repeat": repeat,
                        "value": value,
                    }
                )
        return rows

    def test_median_and_spread(self) -> None:
        summary = perf_suite.summarize(self.rows())
        self.assertEqual([item["metric"] for item in summary], ["p99", "throughput"])
        throughput = next(item for item in summary if item["metric"] == "throughput")
        self.assertAlmostEqual(throughput["median"], 100.0)
        self.assertEqual(throughput["samples"], 3)
        self.assertAlmostEqual(throughput["spread_pct"], 4.0)

    def test_even_median(self) -> None:
        self.assertAlmostEqual(perf_suite.median([1.0, 2.0, 3.0, 4.0]), 2.5)


class EvaluationTest(unittest.TestCase):
    @staticmethod
    def observed(value: float) -> dict:
        return {"median": value, "min": value, "max": value, "samples": 3}

    def test_no_baseline_is_not_a_regression(self) -> None:
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput"), observed=self.observed(1.0),
            baseline=None, floor_pct=5.0,
        )
        self.assertEqual(row["status"], "no-baseline")

    def test_higher_is_better_regression(self) -> None:
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput", "higher"),
            observed=self.observed(90.0),
            baseline={"median": 100.0}, floor_pct=5.0,
        )
        self.assertEqual(row["status"], "regressed")

    def test_higher_within_floor_is_ok(self) -> None:
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput", "higher"),
            observed=self.observed(96.0),
            baseline={"median": 100.0}, floor_pct=5.0,
        )
        self.assertEqual(row["status"], "ok")

    def test_lower_is_better_regression(self) -> None:
        row = perf_suite.evaluate(
            case=None, metric=metric("p99", "lower"),
            observed=self.observed(120.0),
            baseline={"median": 100.0}, floor_pct=10.0,
        )
        self.assertEqual(row["status"], "regressed")

    def test_lower_within_floor_is_ok(self) -> None:
        row = perf_suite.evaluate(
            case=None, metric=metric("p99", "lower"),
            observed=self.observed(108.0),
            baseline={"median": 100.0}, floor_pct=10.0,
        )
        self.assertEqual(row["status"], "ok")

    def test_clock_mismatch_blocks_comparison(self) -> None:
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput", "higher"),
            observed=self.observed(50.0),
            baseline={"median": 100.0, "cpu_mhz_max": 4000.0},
            floor_pct=5.0,
            env={"cpu_mhz_max": 2700.0},
        )
        self.assertEqual(row["status"], "env-mismatch")
        self.assertIsNone(row["ratio"])

    def test_spread_wider_than_floor_is_unstable_not_regressed(self) -> None:
        observed = {"median": 80.0, "min": 60.0, "max": 100.0, "spread_pct": 50.0, "samples": 3}
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput", "higher"),
            observed=observed, baseline={"median": 100.0}, floor_pct=5.0,
        )
        self.assertEqual(row["status"], "unstable")
        self.assertAlmostEqual(row["ratio"], 0.8)

    def test_single_sample_is_never_unstable(self) -> None:
        observed = {"median": 80.0, "min": 80.0, "max": 80.0, "spread_pct": 0.0, "samples": 1}
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput", "higher"),
            observed=observed, baseline={"median": 100.0}, floor_pct=5.0,
        )
        self.assertEqual(row["status"], "regressed")

    def test_noisy_run_without_baseline_is_unstable_not_no_baseline(self) -> None:
        observed = {"median": 80.0, "min": 60.0, "max": 100.0, "spread_pct": 50.0, "samples": 3}
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput", "higher"),
            observed=observed, baseline=None, floor_pct=5.0,
        )
        self.assertEqual(row["status"], "unstable")
        self.assertIsNone(row["ratio"])

    def test_baseline_mismatch_is_not_gated(self) -> None:
        row = perf_suite.evaluate(
            case=None, metric=metric("throughput", "higher"),
            observed=self.observed(50.0),
            baseline={"median": 100.0}, floor_pct=5.0,
            baseline_mismatch="workload parameters changed",
        )
        self.assertEqual(row["status"], "baseline-mismatch")
        self.assertIsNone(row["ratio"])
        self.assertIsNone(row["baseline"])

    def test_baseline_mismatch_detects_workload_change(self) -> None:
        case = perf_suite.Case(
            id="case-cmd", layer="storage", runner="kvengine", labels=("perf",),
            repeats=1, noise_floor_pct=5.0, params={"thread_num": 4},
            metrics=(metric("throughput"),),
        )
        env = {"cmake_cache": {"CMAKE_BUILD_TYPE": "Release"}}
        matching = {"resolution_hint": {"thread_num": 4}, "cmake_cache": env["cmake_cache"]}
        self.assertIsNone(perf_suite.baseline_mismatch(matching, case, env))
        changed_params = {"resolution_hint": {"thread_num": 64}, "cmake_cache": env["cmake_cache"]}
        self.assertEqual(
            perf_suite.baseline_mismatch(changed_params, case, env),
            "workload parameters changed",
        )
        changed_build = {"resolution_hint": {"thread_num": 4}, "cmake_cache": {}}
        self.assertEqual(
            perf_suite.baseline_mismatch(changed_build, case, env),
            "cmake cache changed",
        )

    def test_reset_failure_marker_clears_a_stale_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            marker = output_dir / "failures.txt"
            marker.write_text("old failure\n", encoding="utf-8")
            perf_suite._reset_failure_marker(output_dir)
            self.assertFalse(marker.exists())
            # Idempotent on a clean directory.
            perf_suite._reset_failure_marker(output_dir)
            self.assertFalse(marker.exists())


class AbTest(unittest.TestCase):
    def test_param_overrides_parse_json_and_keep_the_rest(self) -> None:
        case = perf_suite.load_cases()[0]
        patched = perf_suite._apply_param_overrides(
            case, ["thread_num=2", "ssd_path={work_dir}/ssd", "zipfian_scramble=true"]
        )
        self.assertEqual(patched.params["thread_num"], 2)
        self.assertIs(patched.params["zipfian_scramble"], True)
        self.assertEqual(patched.params["ssd_path"], "{work_dir}/ssd")
        for key, value in case.params.items():
            if key not in {"thread_num", "ssd_path", "zipfian_scramble"}:
                self.assertEqual(patched.params[key], value)
        # The registry itself must stay untouched (Case is frozen, params copied).
        self.assertNotEqual(case.params.get("thread_num"), 2)

    def test_param_override_requires_key_value(self) -> None:
        case = perf_suite.load_cases()[0]
        with self.assertRaises(perf_suite.SuiteError):
            perf_suite._apply_param_overrides(case, ["thread_num"])

    def test_spread_pct_matches_summarize(self) -> None:
        self.assertAlmostEqual(perf_suite._spread_pct([100.0, 102.0, 98.0]), 4.0)
        self.assertEqual(perf_suite._spread_pct([5.0]), 0.0)

    def test_large_effect_with_noisy_sides_is_still_regressed(self) -> None:
        # Disjoint ranges: a 3x slowdown stays a result at 15% spread.
        ratio, status = perf_suite.ab_status(
            "higher", [1.90e6, 1.91e6], [5.53e5, 6.41e5], 5.0
        )
        self.assertAlmostEqual(ratio, 0.313, places=2)
        self.assertEqual(status, "regressed")

    def test_small_effect_with_overlapping_ranges_is_unstable(self) -> None:
        ratio, status = perf_suite.ab_status(
            "higher", [100.0, 98.0, 102.0], [94.0, 99.0, 93.0], 5.0
        )
        self.assertAlmostEqual(ratio, 0.94, places=2)
        self.assertEqual(status, "unstable")

    def test_improvement_is_never_a_regression(self) -> None:
        ratio, status = perf_suite.ab_status(
            "higher", [100.0, 101.0, 99.0], [140.0, 141.0, 139.0], 5.0
        )
        self.assertEqual(status, "ok")
        _, status = perf_suite.ab_status(
            "lower", [100.0, 101.0, 99.0], [80.0, 81.0, 79.0], 5.0
        )
        self.assertEqual(status, "ok")


class RegistryTest(unittest.TestCase):
    def test_committed_registry_is_valid(self) -> None:
        cases = perf_suite.load_cases()
        self.assertTrue(cases)
        ids = {case.id for case in cases}
        self.assertIn("kvengine_c_read", ids)
        smoke = [case for case in cases if "perf_smoke" in case.labels]
        self.assertGreaterEqual(len(smoke), 3)
        for case in cases:
            for spec in case.metrics:
                self.assertIn(spec.direction, {"higher", "lower"})

    def test_tiered_case_keeps_scramble(self) -> None:
        case = next(
            item
            for item in perf_suite.load_cases()
            if item.id == "kvengine_tiered_scramble"
        )
        self.assertIs(case.params["zipfian_scramble"], True)
        self.assertEqual(case.params["ssd_path"], "{work_dir}/ssd")
        # Placement-only row: the cache is measured with `ab`, not baked in here.
        self.assertEqual(case.params["tiering_cache_policy"], "none")

    def test_selection_rejects_unknown_id(self) -> None:
        with self.assertRaises(perf_suite.SuiteError):
            perf_suite.select_cases(perf_suite.load_cases(), ["nope"], None)

    def test_baseline_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = perf_suite.BASELINE_ROOT
            perf_suite.BASELINE_ROOT = Path(tmp)
            try:
                case = perf_suite.Case(
                    id="case-a",
                    layer="storage",
                    runner="kvengine",
                    labels=("perf",),
                    repeats=3,
                    noise_floor_pct=5.0,
                    params={},
                    metrics=(metric("throughput"),),
                )
                rows = [
                    {"case": "case-a", "metric": "throughput", "median": 100.0,
                     "min": 99.0, "max": 101.0, "spread_pct": 2.0, "samples": 3}
                ]
                path = perf_suite.write_baseline("env-x", case, rows, {"git_commit": "abc"})
                self.assertTrue(path.exists())
                loaded = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(loaded["metrics"]["throughput"]["median"], 100.0)
                self.assertEqual(perf_suite.load_baseline("env-x", "case-a")["case"], "case-a")
                self.assertIsNone(perf_suite.load_baseline("env-x", "missing"))
            finally:
                perf_suite.BASELINE_ROOT = original


if __name__ == "__main__":
    unittest.main()
