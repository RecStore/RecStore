from __future__ import annotations

import unittest

from model_zoo.rs_demo.config import parse_config


class TestHybridConfig(unittest.TestCase):
    def test_parse_config_accepts_dlrm_arch_flags(self) -> None:
        cfg = parse_config(
            [
                "--embedding-dim",
                "128",
                "--dense-arch-layer-sizes",
                "512,256,128",
                "--over-arch-layer-sizes",
                "1024,1024,512,256,1",
            ]
        )

        self.assertEqual(cfg.embedding_dim, 128)
        self.assertEqual(cfg.dense_arch_layer_sizes, "512,256,128")
        self.assertEqual(cfg.over_arch_layer_sizes, "1024,1024,512,256,1")


if __name__ == "__main__":
    unittest.main()
