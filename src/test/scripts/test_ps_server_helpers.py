import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tools.config.recstore_config_path as recstore_config_path
import ps_server_helpers


class TestPSServerHelpers(unittest.TestCase):
    def test_find_config_file_prefers_default_docker_path_from_nested_cwd(self):
        with tempfile.TemporaryDirectory() as default_tmp, tempfile.TemporaryDirectory() as cwd_tmp:
            default_config = Path(default_tmp) / "recstore_config.json"
            nested = Path(cwd_tmp) / "a" / "b"
            nested.mkdir(parents=True)
            default_config.write_text("{}", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                recstore_config_path,
                "DEFAULT_RECSTORE_CONFIG_PATH",
                default_config,
            ), mock.patch.object(ps_server_helpers.os, "getcwd", return_value=str(nested)):
                self.assertEqual(ps_server_helpers.find_config_file(), str(default_config))

    def test_get_rdma_skip_reason_when_infiniband_dir_missing(self):
        with mock.patch.object(ps_server_helpers.os.path, "isdir", return_value=False):
            reason = ps_server_helpers.get_rdma_skip_reason()

        self.assertIn("/dev/infiniband", reason)

    def test_get_rdma_skip_reason_returns_none_when_uverbs_device_exists(self):
        with mock.patch.object(ps_server_helpers.os.path, "isdir", return_value=True):
            with mock.patch.object(
                ps_server_helpers.glob,
                "glob",
                return_value=["/dev/infiniband/uverbs0"],
            ):
                reason = ps_server_helpers.get_rdma_skip_reason()

        self.assertIsNone(reason)

    def test_should_skip_server_start_raises_when_ports_partially_open(self):
        with mock.patch.object(
            ps_server_helpers,
            "run_launcher_decision",
            return_value={
                "should_start": False,
                "should_fail": True,
                "reason": "ps_server ports are partially available: expected=[15000,15001], open=[15000]",
                "configured_ports": [15000, 15001],
                "open_ports": [15000],
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "partially available"):
                ps_server_helpers.should_skip_server_start()


if __name__ == '__main__':
    unittest.main()
