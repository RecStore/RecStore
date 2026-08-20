import json
import os
import tempfile
import unittest
from unittest import mock

from ..optimizer import SparseAdamW, SparseRowWiseAdagrad


class _FakeRecStoreModule:
    def __init__(self) -> None:
        self._config_names = {"feature": "table"}
        self._trace = []
        self.kv_client = object()


class SparseOptimizerConfigTest(unittest.TestCase):
    def _config_path(self, optimizer: dict) -> str:
        config_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        )
        self.addCleanup(lambda: os.unlink(config_file.name))
        with config_file:
            json.dump({"cache_ps": {"optimizer": optimizer}}, config_file)
        return config_file.name

    def test_accepts_matching_rowwise_adagrad_config(self) -> None:
        path = self._config_path(
            {
                "type": "RowWiseAdagrad",
                "learning_rate": 0.001,
                "epsilon": 1e-8,
            }
        )
        with mock.patch.dict(os.environ, {"RECSTORE_CONFIG": path}):
            optimizer = SparseRowWiseAdagrad(
                [_FakeRecStoreModule()], lr=0.001, eps=1e-8
            )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(optimizer.eps, 1e-8)

    def test_rejects_server_optimizer_mismatch(self) -> None:
        path = self._config_path(
            {"type": "SGD", "learning_rate": 0.01}
        )
        with mock.patch.dict(os.environ, {"RECSTORE_CONFIG": path}):
            with self.assertRaisesRegex(
                RuntimeError, "RecStore sparse optimizer mismatch"
            ):
                SparseRowWiseAdagrad(
                    [_FakeRecStoreModule()], lr=0.001, eps=1e-8
                )

    def test_accepts_matching_adamw_config(self) -> None:
        path = self._config_path(
            {
                "type": "AdamW",
                "learning_rate": 0.001,
                "epsilon": 1e-8,
                "beta1": 0.9,
                "beta2": 0.98,
                "weight_decay": 0.0,
            }
        )
        with mock.patch.dict(os.environ, {"RECSTORE_CONFIG": path}):
            optimizer = SparseAdamW(
                [_FakeRecStoreModule()],
                lr=0.001,
                betas=(0.9, 0.98),
                eps=1e-8,
                weight_decay=0.0,
            )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(optimizer.betas, (0.9, 0.98))


if __name__ == "__main__":
    unittest.main()
