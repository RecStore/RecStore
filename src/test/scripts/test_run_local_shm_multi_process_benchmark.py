import unittest
from pathlib import Path

from tools.benchmarks.run_local_shm_multi_process_benchmark import (  # noqa: E402
    build_worker_env,
    build_worker_label,
    resolve_ready_queue_index,
)


class TestRunLocalShmMultiProcessBenchmark(unittest.TestCase):
    def test_build_worker_env_sets_ready_queue_index(self):
        env = build_worker_env(worker_rank=3, base_env={"PATH": "/usr/bin"})
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["RECSTORE_LOCAL_SHM_READY_QUEUE_INDEX"], "3")

    def test_build_worker_label_uses_worker_rank(self):
        self.assertEqual(build_worker_label(0), "worker0")
        self.assertEqual(build_worker_label(7), "worker7")

    def test_resolve_ready_queue_index_wraps_by_queue_count(self):
        self.assertEqual(resolve_ready_queue_index(0, 2), 0)
        self.assertEqual(resolve_ready_queue_index(1, 2), 1)
        self.assertEqual(resolve_ready_queue_index(2, 2), 0)
        self.assertEqual(resolve_ready_queue_index(5, 4), 1)


if __name__ == "__main__":
    unittest.main()
