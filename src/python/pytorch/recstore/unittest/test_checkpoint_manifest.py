import json
import shutil
import tempfile
import unittest
from pathlib import Path

from recstore.checkpoint import (
    MANIFEST_NAME,
    CheckpointError,
    PeriodicSparseCheckpoint,
    committed_checkpoints,
    latest_committed,
    save_sparse_checkpoint,
)


class FakeClient:
    """Writes the shard files the dist client would write, or fails on demand."""

    def __init__(self, shards=2, fail=False):
        self.shards = shards
        self.fail = fail
        self.calls = []

    def save_checkpoint(self, path, metadata):
        directory = Path(path)
        self.calls.append((directory, metadata))
        if self.fail:
            raise RuntimeError("injected save failure")
        for shard in range(self.shards):
            (directory / f"recstore-shard-{shard}.bin").write_bytes(
                bytes([shard + 1]) * (100 + shard)
            )


IDENTITY = {"schema_version": 1, "table": {"name": "t_a", "dim": 8}}


class CheckpointManifestTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="recstore-ckpt-test-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_save_commits_manifest_after_shards(self):
        client = FakeClient(shards=3)
        checkpoint = save_sparse_checkpoint(
            client, self.root, step=5, identity=IDENTITY, keep=2
        )

        self.assertEqual(checkpoint.checkpoint_id, "step-00000005")
        self.assertEqual(
            checkpoint.files,
            (
                ("recstore-shard-0.bin", 100),
                ("recstore-shard-1.bin", 101),
                ("recstore-shard-2.bin", 102),
            ),
        )
        manifest = json.loads((checkpoint.directory / MANIFEST_NAME).read_text())
        self.assertEqual(manifest["checkpoint_id"], "step-00000005")
        self.assertEqual(manifest["step"], 5)
        self.assertEqual(manifest["identity"], IDENTITY)
        self.assertEqual([item["name"] for item in manifest["files"]], [
            "recstore-shard-0.bin",
            "recstore-shard-1.bin",
            "recstore-shard-2.bin",
        ])
        # The client receives the base metadata; the PS adds shard_id itself.
        self.assertEqual(
            client.calls[0][1],
            {"checkpoint_id": "step-00000005", "identity": IDENTITY},
        )
        self.assertFalse(any(p.name.startswith(".") for p in self.root.iterdir()))
        self.assertEqual(latest_committed(self.root)["step"], 5)

    def test_failed_save_is_not_committed(self):
        with self.assertRaises(RuntimeError):
            save_sparse_checkpoint(
                FakeClient(fail=True), self.root, step=5, identity=IDENTITY
            )

        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIsNone(latest_committed(self.root))

    def test_save_without_shards_is_rejected(self):
        with self.assertRaises(CheckpointError):
            save_sparse_checkpoint(
                FakeClient(shards=0), self.root, step=5, identity=IDENTITY
            )
        self.assertIsNone(latest_committed(self.root))

    def test_retention_keeps_newest(self):
        client = FakeClient()
        for step in (5, 10, 15):
            save_sparse_checkpoint(
                client, self.root, step=step, identity=IDENTITY, keep=2
            )

        self.assertEqual(
            [manifest["step"] for manifest, _ in committed_checkpoints(self.root)],
            [15, 10],
        )
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()),
            ["step-00000010", "step-00000015"],
        )

    def test_failed_resave_keeps_previous_version(self):
        save_sparse_checkpoint(FakeClient(), self.root, step=5, identity=IDENTITY)
        original = json.loads(
            (self.root / "step-00000005" / MANIFEST_NAME).read_text()
        )

        with self.assertRaises(RuntimeError):
            save_sparse_checkpoint(
                FakeClient(fail=True), self.root, step=5, identity=IDENTITY
            )

        manifest = latest_committed(self.root)
        self.assertEqual(manifest["files"], original["files"])
        self.assertEqual(
            sorted(path.name for path in (self.root / "step-00000005").iterdir()),
            ["manifest.json", "recstore-shard-0.bin", "recstore-shard-1.bin"],
        )

    def test_latest_ignores_directory_without_manifest(self):
        save_sparse_checkpoint(FakeClient(), self.root, step=5, identity=IDENTITY)
        half = self.root / "step-00000009"
        half.mkdir()
        (half / "recstore-shard-0.bin").write_bytes(b"x")

        manifests = committed_checkpoints(self.root)
        self.assertEqual([manifest["step"] for manifest, _ in manifests], [5])

    def test_periodic_saves_only_on_step_boundary(self):
        checkpointer = PeriodicSparseCheckpoint(
            FakeClient(),
            self.root / "periodic",
            IDENTITY,
            every_n_steps=5,
            keep=2,
        )

        self.assertIsNone(checkpointer.maybe_save(0))
        self.assertIsNone(checkpointer.maybe_save(3))
        self.assertEqual(checkpointer.maybe_save(5).step, 5)
        self.assertIsNone(checkpointer.maybe_save(6))
        self.assertEqual(checkpointer.maybe_save(10).step, 10)
        self.assertEqual([saved.step for saved in checkpointer.saved], [5, 10])


if __name__ == "__main__":
    unittest.main()
