"""Sparse (PS) checkpoint layout: one directory per version plus a manifest.

A version is committed only when ``manifest.json`` exists inside its directory.
The manifest is written after the client has saved every shard, and the
directory only appears under its final name after the manifest is in place, so
an interrupted save can never produce something a reader would treat as a
complete checkpoint.

The manifest stores the identity separately from ``checkpoint_id``; the PS
server re-serialises both together with its own ``shard_id`` before comparing
them, so key order and whitespace do not matter.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from . import __version__

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
SHARD_GLOB = "recstore-shard-*.bin"
STAGING_PREFIX = ".staging-"

__all__ = [
    "CheckpointError",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "PeriodicSparseCheckpoint",
    "SparseCheckpoint",
    "checkpoint_id_for_step",
    "latest_committed",
    "save_sparse_checkpoint",
]


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be written or read back."""


@dataclass(frozen=True)
class SparseCheckpoint:
    checkpoint_id: str
    directory: Path
    step: int
    files: tuple[tuple[str, int], ...]


def checkpoint_id_for_step(step: int) -> str:
    if step <= 0:
        raise ValueError(f"checkpoint step must be positive, got {step}")
    return f"step-{int(step):08d}"


def _checkpoint_metadata(
    checkpoint_id: str, identity: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("checkpoint_id must be a non-empty string")
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("checkpoint identity must be a non-empty mapping")
    return {"checkpoint_id": checkpoint_id, "identity": dict(identity)}


def _shard_files(directory: Path) -> tuple[tuple[str, int], ...]:
    files = sorted(
        (path.name, path.stat().st_size)
        for path in directory.glob(SHARD_GLOB)
        if path.is_file()
    )
    if not files:
        raise CheckpointError(f"no {SHARD_GLOB} files were written to {directory}")
    return tuple(files)


def _write_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    target = directory / MANIFEST_NAME
    staging = directory / (MANIFEST_NAME + ".tmp")
    staging.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(staging, target)


def save_sparse_checkpoint(
    client: Any,
    root: os.PathLike[str] | str,
    *,
    step: int,
    identity: Mapping[str, Any],
    checkpoint_id: Optional[str] = None,
    keep: int = 2,
    extra: Optional[Mapping[str, Any]] = None,
) -> SparseCheckpoint:
    """Save every shard under ``root/<checkpoint_id>/`` and commit it.

    ``client`` only needs a ``save_checkpoint(path, metadata)`` method; the
    in-tree ``RecStoreClient`` provides it. Raises :class:`CheckpointError`
    when the client reports nothing was written.
    """
    root = Path(root)
    checkpoint_id = checkpoint_id or checkpoint_id_for_step(step)
    metadata = _checkpoint_metadata(checkpoint_id, identity)
    final = root / checkpoint_id
    staging = root / f"{STAGING_PREFIX}{checkpoint_id}-{os.getpid()}"

    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    try:
        client.save_checkpoint(str(staging), metadata)
        files = _shard_files(staging)
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "step": int(step),
            "identity": dict(identity),
            "files": [{"name": name, "size": size} for name, size in files],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recstore_version": __version__,
        }
        if extra:
            manifest["extra"] = dict(extra)
        _write_manifest(staging, manifest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Re-saving the same step replaces the old version only after the new one
    # is complete.
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    _prune(root, keep)
    return SparseCheckpoint(
        checkpoint_id=checkpoint_id, directory=final, step=int(step), files=files
    )


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("checkpoint_id"), str)
        or not isinstance(manifest.get("step"), int)
        or not isinstance(manifest.get("identity"), dict)
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        return None
    return manifest


def committed_checkpoints(
    root: os.PathLike[str] | str,
) -> list[tuple[dict[str, Any], Path]]:
    """Return ``(manifest, directory)`` pairs, newest step first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[tuple[dict[str, Any], Path]] = []
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        manifest = _read_manifest(directory / MANIFEST_NAME)
        if manifest is not None:
            found.append((manifest, directory))
    found.sort(key=lambda item: item[0]["step"], reverse=True)
    return found


def latest_committed(root: os.PathLike[str] | str) -> Optional[dict[str, Any]]:
    """Newest committed manifest under ``root``, or ``None``."""
    found = committed_checkpoints(root)
    return found[0][0] if found else None


def _prune(root: Path, keep: int) -> None:
    keep = max(0, int(keep))
    for _, directory in committed_checkpoints(root)[keep:]:
        shutil.rmtree(directory, ignore_errors=True)


class PeriodicSparseCheckpoint:
    """Save a committed version every ``every_n_steps`` at a step boundary.

    Rank 0 owns the checkpointer; the other ranks must not call it. A save
    blocks the PS engine's writers while the snapshot is dumped, so call it
    after the step's sparse update has been submitted, not in the middle of a
    step.
    """

    def __init__(
        self,
        client: Any,
        root: os.PathLike[str] | str,
        identity: Mapping[str, Any],
        *,
        every_n_steps: int,
        keep: int = 2,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        if keep < 1:
            raise ValueError("keep must be at least 1")
        self._client = client
        self._root = Path(root)
        self._identity = dict(identity)
        self.every_n_steps = int(every_n_steps)
        self.keep = int(keep)
        self.saved: list[SparseCheckpoint] = []

    def maybe_save(self, step: int) -> Optional[SparseCheckpoint]:
        """Save when ``step`` is a positive multiple of ``every_n_steps``."""
        if step <= 0 or step % self.every_n_steps != 0:
            return None
        checkpoint = save_sparse_checkpoint(
            self._client,
            self._root,
            step=step,
            identity=self._identity,
            keep=self.keep,
        )
        self.saved.append(checkpoint)
        return checkpoint
