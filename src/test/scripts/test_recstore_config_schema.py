import json
import unittest
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "tools" / "config" / "recstore_config.schema.json"


class TestRecstoreConfigSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def validate_config(self, base_kv_config: dict, optimizer: dict | None = None) -> None:
        config = {
            "cache_ps": {
                "ps_type": "GRPC",
                "max_batch_keys_size": 1024,
                "num_threads": 4,
                "num_shards": 1,
                "servers": [
                    {"host": "127.0.0.1", "port": 15000, "shard": 0},
                ],
                "base_kv_config": base_kv_config,
            },
            "distributed_client": {
                "num_shards": 1,
                "hash_method": "city_hash",
                "servers": [
                    {"host": "127.0.0.1", "port": 15000, "shard": 0},
                ],
            },
            "client": {"host": "127.0.0.1", "port": 15000, "shard": 0},
        }
        if optimizer is not None:
            config["cache_ps"]["optimizer"] = optimizer
        jsonschema.Draft202012Validator(self.schema).validate(config)

    def assert_invalid_config(self, base_kv_config: dict) -> None:
        config = {
            "cache_ps": {
                "ps_type": "GRPC",
                "max_batch_keys_size": 1024,
                "num_threads": 4,
                "num_shards": 1,
                "servers": [
                    {"host": "127.0.0.1", "port": 15000, "shard": 0},
                ],
                "base_kv_config": base_kv_config,
            },
            "distributed_client": {
                "num_shards": 1,
                "hash_method": "city_hash",
                "servers": [
                    {"host": "127.0.0.1", "port": 15000, "shard": 0},
                ],
            },
            "client": {"host": "127.0.0.1", "port": 15000, "shard": 0},
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(self.schema).validate(config)

    def test_accepts_nested_local_kv_config_without_engine_type(self) -> None:
        self.validate_config(
            {
                "capacity": 1024,
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "default_value_size_hint": 128,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": 1024 * 128,
                    },
                },
            }
        )

    def test_accepts_rowwise_adagrad_optimizer(self) -> None:
        self.validate_config(
            {
                "capacity": 1024,
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "default_value_size_hint": 128,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": 1024 * 128,
                    },
                },
            },
            optimizer={
                "type": "RowWiseAdagrad",
                "learning_rate": 0.001,
                "epsilon": 1e-8,
            },
        )

    def test_rejects_unknown_sparse_optimizer(self) -> None:
        with self.assertRaises(jsonschema.ValidationError):
            self.validate_config(
                {},
                optimizer={"type": "Adam", "learning_rate": 0.001},
            )

    def test_rejects_legacy_flat_kv_config(self) -> None:
        self.assert_invalid_config(
            {
                "engine_type": "KVEngineComposite",
                "path": "/tmp/recstore_data",
                "capacity": 1024,
                "index_type": "DRAM",
                "value_type": "SSD",
                "value_size": 128,
                "value_memory_management": "PersistLoopShmMalloc",
                "queue_size": 64,
            }
        )

    def test_accepts_petkv_config(self) -> None:
        self.validate_config(
            {
                "engine_type": "KVEnginePetKV",
                "path": "/dev/shm/recstore_petkv",
                "capacity": 1000000,
                "value_size": 512,
                "value_capacity": 536870912,
                "shard_num": 16,
            }
        )

    def test_accepts_external_fasterkv_config(self) -> None:
        self.validate_config(
            {
                "engine_type": "KVEngineFasterKV",
                "path": "/tmp/fasterkv_data",
                "capacity": 1024,
                "value_size": 128,
                "max_batch_size": 64,
            }
        )

    def test_accepts_external_fasterkv_ssd_config(self) -> None:
        base_kv_properties = self.schema["$defs"]["baseKvConfig"]["properties"]
        self.assertIn("fasterkv", base_kv_properties)

        self.validate_config(
            {
                "engine_type": "KVEngineFasterKV",
                "path": "/tmp/fasterkv_data",
                "capacity": 1000000,
                "value_size": 512,
                "fasterkv": {
                    "storage": "ssd",
                    "log_path": "/data/fasterkv/hlog",
                    "hlog_memory_bytes": 1073741824,
                    "mutable_fraction": 0.9,
                    "read_cache_bytes": 268435456,
                },
            }
        )

    def test_accepts_external_hps_rocksdb_config(self) -> None:
        self.validate_config(
            {
                "engine_type": "KVEngineHPSRocksDB",
                "path": "/tmp/hps_data",
                "rocksdb_path": "/tmp/hps_rocksdb",
                "capacity": 1024,
                "value_size": 128,
                "table_name": "default",
            }
        )

    def test_rejects_removed_external_engine_type_field(self) -> None:
        self.assert_invalid_config(
            {
                "external_engine_type": "KVEngineFasterKV",
                "engine_type": "KVEngineFasterKV",
                "path": "/tmp/fasterkv_data",
                "capacity": 1024,
                "value_size": 128,
            }
        )


if __name__ == "__main__":
    unittest.main()
