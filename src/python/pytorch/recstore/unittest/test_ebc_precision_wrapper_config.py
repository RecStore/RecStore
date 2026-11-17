import os
import sys
import unittest
import json

UNITTEST_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_ROOT = os.path.abspath(os.path.join(UNITTEST_DIR, '../..'))
REPO_ROOT = os.path.abspath(os.path.join(UNITTEST_DIR, '../../../../..'))
if PYTORCH_ROOT not in sys.path:
    sys.path.insert(0, PYTORCH_ROOT)

from recstore.unittest import test_ebc_precision_wrapper as wrapper


class TestEBCPrecisionWrapperConfig(unittest.TestCase):
    def test_resolve_repo_config_path_points_to_repo_root(self):
        config_path = wrapper._resolve_repo_config_path()
        self.assertEqual(config_path, os.path.join(REPO_ROOT, "recstore_config.json"))
        self.assertTrue(os.path.exists(config_path))

    def test_resolve_ps_endpoint_uses_repo_config(self):
        config_path = wrapper._resolve_repo_config_path()
        with open(config_path, "r") as f:
            config = json.load(f)

        expected_host = config["client"]["host"]
        expected_port = config["client"]["port"]

        host, port = wrapper._resolve_ps_endpoint()
        self.assertEqual(host, expected_host)
        self.assertEqual(port, expected_port)
