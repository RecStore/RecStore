import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_petps_fault_injection import FAULT_SHARD, SCENARIOS, SERVER_COUNT

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SOURCE = REPO_ROOT / "src/test/ps/rdma/petps_integration_test.cpp"


class TestRunPetPSFaultInjection(unittest.TestCase):
    """Keep the driver's contract with the C++ test bodies from drifting.

    The scenarios can only run on a host with RDMA hardware, so nothing in CI
    exercises them; these checks are the automated guard on the marker names,
    env vars and gtest filters the driver has to agree on.
    """

    def setUp(self):
        self.source = TEST_SOURCE.read_text(encoding="utf-8")

    def test_covers_the_three_gated_scenarios(self):
        self.assertEqual(sorted(SCENARIOS), ["fast_fail", "killed", "rebuild"])

    def test_every_scenario_targets_a_test_in_the_source(self):
        for name, spec in SCENARIOS.items():
            test_name = spec.gtest_filter.split(".")[-1]
            with self.subTest(scenario=name):
                self.assertTrue(spec.gtest_filter.startswith("PetPSIntegrationTest."))
                self.assertIn(f"TEST(PetPSIntegrationTest, {test_name})", self.source)
                self.assertIn(f'getenv("{spec.env_var}")', self.source)
                self.assertIn(spec.warm_marker, self.source)

    def test_freeze_until_dirty_scenarios_wait_for_a_dirty_marker(self):
        for name, spec in SCENARIOS.items():
            with self.subTest(scenario=name):
                if spec.fault == "freeze_until_dirty":
                    self.assertIsNotNone(spec.dirty_marker)
                    self.assertIn(spec.dirty_marker, self.source)

    def test_every_scenario_has_more_than_one_lane(self):
        for name, spec in SCENARIOS.items():
            with self.subTest(scenario=name):
                self.assertGreater(spec.lanes, 1)

    def test_fault_shard_is_not_the_control_plane_shard(self):
        self.assertGreater(FAULT_SHARD, 0)
        self.assertLess(FAULT_SHARD, SERVER_COUNT)


if __name__ == "__main__":
    unittest.main()
