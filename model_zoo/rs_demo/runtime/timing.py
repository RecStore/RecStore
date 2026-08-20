from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


@contextmanager
def stage_timer(row: dict[str, Any], key: str):
    """Wall-clock timer for a single stage.

    Kept for callers that manage their own device synchronization.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        row[key] = (time.perf_counter() - start) * 1e3


_VALID_MODES = ("stage", "step", "none")


class StepTimer:
    """Per-stage timing for one training step.

    GPU-compute stages (`gpu`) are measured with CUDA events and resolved in
    `finish` after a single device drain, so each stage captures only its own
    GPU work and never absorbs a neighbor's un-drained tail. Host or network
    stages (`cpu`) use the wall clock, because CUDA events read ~0 for work that
    never touches the compute stream (dataloader, PS/RDMA round trips). On
    non-CUDA devices `gpu` falls back to the wall clock.

    Modes:
        stage: Per-stage CUDA events, single sync in `finish` (default).
        step:  No per-stage CUDA events (wall-clock), single sync in `finish`.
        none:  No per-stage events, no explicit sync.
    """

    def __init__(self, row: dict[str, Any], torch, device, *, mode: str = "stage") -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid timing sync mode {mode!r}; must be one of {', '.join(_VALID_MODES)}"
            )
        self._row = row
        self._torch = torch
        self._device = device
        self._cuda = device.type == "cuda"
        self._mode = mode
        self._pending: list[tuple[str, Any, Any]] = []

    @contextmanager
    def cpu(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._row[key] = (time.perf_counter() - start) * 1e3

    @contextmanager
    def gpu(self, key: str):
        # "stage" mode: per-stage CUDA events on GPU; wall-clock fallback otherwise.
        # "step"/"none": always wall-clock, no per-stage events.
        if not self._cuda or self._mode != "stage":
            with self.cpu(key):
                yield
            return
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((key, start, end))

    def finish(self) -> float:
        """Drain the device once, resolve GPU stage timings, return the drain wait (ms).

        The drain wait is the time the host blocks for outstanding GPU work and
        collectives, i.e. the cross-rank straggler cost.

        "none" mode: no explicit sync, pending events discarded.
        "step"/"stage" mode: single device sync, then resolve events.
        """
        if self._mode == "none":
            self._pending.clear()
            return 0.0
        if not self._cuda:
            return 0.0
        wait_start = time.perf_counter()
        self._torch.cuda.synchronize(self._device)
        wait_ms = (time.perf_counter() - wait_start) * 1e3
        for key, start, end in self._pending:
            self._row[key] = start.elapsed_time(end)
        self._pending.clear()
        return wait_ms
