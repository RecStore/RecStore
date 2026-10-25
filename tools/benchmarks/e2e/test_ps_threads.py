import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.benchmarks.e2e.custom.config import (  # noqa: E402
    BenchmarkConfig,
    ClientSpec,
    ServerSpec,
)
from tools.benchmarks.e2e.custom import runtime  # noqa: E402


class TestResolvePsThreads(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = BenchmarkConfig(clients=(ClientSpec(),), servers=(ServerSpec(),))
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("RECSTORE_E2E_RDMA_SERVER_THREADS", None)

    def tearDown(self) -> None:
        self._env.stop()

    def _auto_with_cores(self, cores: int) -> int:
        with mock.patch.object(runtime, "_physical_cores", lambda *args, **kwargs: cores):
            return runtime.resolve_ps_threads(self.cfg)

    def test_auto_leaves_most_cores_to_the_trainer(self):
        # 9 physical / 18 logical is the host from #295, where 16 threads lost
        # ~55% throughput to oversubscription.
        self.assertEqual(self._auto_with_cores(9), 2)

    def test_auto_is_clamped(self):
        self.assertEqual(self._auto_with_cores(1), 1)
        self.assertEqual(self._auto_with_cores(2), 1)
        self.assertEqual(self._auto_with_cores(200), 16)

    def test_explicit_flag_beats_env(self):
        os.environ["RECSTORE_E2E_RDMA_SERVER_THREADS"] = "3"
        self.assertEqual(runtime.resolve_ps_threads(replace(self.cfg, ps_threads=7)), 7)

    def test_env_beats_auto(self):
        os.environ["RECSTORE_E2E_RDMA_SERVER_THREADS"] = "3"
        self.assertEqual(runtime.resolve_ps_threads(self.cfg), 3)


if __name__ == "__main__":
    unittest.main()
