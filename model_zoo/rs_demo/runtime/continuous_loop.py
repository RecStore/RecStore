"""Experimental all-step timing; terminal events do not replace PS global ACK."""
from __future__ import annotations

import math
import os
import time
from collections import deque


COMPLETION_COLUMNS = (
    "step_total_ms", "step_end_to_end_ms", "step_wall_ms",
    "samples_per_sec", "batches_per_sec",
)


def verify_continuous_protocol(torch, dist, device, world_size, steps):
    """All ranks participate, including OFF ranks; this is setup, not training."""
    if world_size <= 1:
        return
    local = torch.tensor([int(os.environ.get("RECSTORE_CONTINUOUS_LOOP") == "1"),
                          int(steps)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    values = [part.cpu().tolist() for part in gathered]
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError("Continuous mode/steps mismatch across ranks: " + repr(values))


def continuous_enabled(cfg, device, world_size, *, lane):
    if os.environ.get("RECSTORE_CONTINUOUS_LOOP") != "1":
        return False
    if (world_size != 2 or device.type != "cuda" or not cfg.dense_cuda_graph
            or cfg.steps <= 0 or cfg.torchrec_dist_mode == "fair_remote"):
        raise ValueError("Continuous loop requires world2 CUDA dense graph and positive steps")
    if (cfg.recstore_profiler or cfg.torchrec_profiler or cfg.recstore_event_log
            or cfg.torchrec_compare_recstore_csv):
        raise ValueError("Continuous loop forbids profilers, event logs and legacy CSV comparison")
    if lane == "recstore":
        if cfg.read_mode != "ps_pipeline" or cfg.ps_type.upper() != "RDMA":
            raise ValueError("Continuous RecStore requires the selected RDMA PS owner pipeline")
    elif lane != "torchrec" or cfg.torchrec_memory_mode != "hbm":
        raise ValueError("Continuous TorchRec requires HBM")
    return True


class ContinuousLoop:
    """Own independent loss slots and at most two event/step pairs, never batches."""

    def __init__(self, torch, device, steps):
        self.torch, self.device, self.steps = torch, device, steps
        self.pending = deque()
        self.written = self.max_pending = 0
        self.started = self.finished = False
        self.losses = self.cpu_losses = None
        self.summary = None

    def start(self):
        if self.started:
            raise RuntimeError("Continuous loop started twice")
        # The graph/setup barrier precedes this call in both runners. Drain its
        # remaining GPU tail before the common all-training interval begins.
        self.torch.cuda.synchronize(self.device)
        self.started = True
        self.start_time = time.perf_counter()  # Includes both loss allocations.
        try:
            self.losses = self.torch.empty(self.steps, dtype=self.torch.float32, device=self.device)
            self.cpu_losses = self.torch.empty(self.steps, dtype=self.torch.float32,
                                               device="cpu", pin_memory=True)
        except BaseException as error:
            self.drain_after_error(error)
            raise
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.drain_after_error(error)
        elif not self.finished:
            error = RuntimeError("Continuous loop exited without the final drain")
            self.drain_after_error(error)
            raise error
        return False

    def admit(self, step):
        if not self.started or self.finished or step != self.written:
            raise RuntimeError("Continuous step order mismatch")
        start = time.perf_counter()
        while self.pending and self.pending[0][1].query():
            self.pending.popleft()
        if len(self.pending) == 2:
            self.pending[0][1].synchronize()
            self.pending.popleft()
        return (time.perf_counter() - start) * 1e3

    def record_loss_and_terminal(self, step, loss):
        if step != self.written or len(self.pending) >= 2:
            raise RuntimeError("Continuous loss/event admission mismatch")
        if loss.device != self.device or loss.dtype != self.torch.float32 or loss.numel() != 1:
            raise RuntimeError("Continuous loss must be a CUDA FP32 scalar")
        # Replay returns fixed storage: copy its bits before the next replay.
        self.losses.narrow(0, step, 1).copy_(loss.detach().reshape(1))
        event = self.torch.cuda.Event(enable_timing=False)
        event.record(self.torch.cuda.current_stream(self.device))
        self.pending.append((step, event))
        self.max_pending = max(self.max_pending, len(self.pending))
        self.written += 1

    def finish_row(self, row, step_start, backpressure_ms):
        row.update({name: "" for name in COMPLETION_COLUMNS})
        row.update(loss="", warmup_excluded=0, step_sync_wait_ms="",
                   host_step_enqueue_wall_ms=(time.perf_counter() - step_start) * 1e3,
                   continuous_backpressure_ms=backpressure_ms,
                   continuous_pending_events=len(self.pending))

    def finish(self, rows, barrier):
        if self.finished or self.written != self.steps or len(rows) != self.steps:
            raise RuntimeError("Continuous loop did not record the exact requested steps")
        if any(int(row["step"]) != step for step, row in enumerate(rows)):
            raise RuntimeError("Continuous output step order mismatch")
        drain_start = time.perf_counter()
        while self.pending:
            self.pending[0][1].synchronize()
            self.pending.popleft()
        self.cpu_losses.copy_(self.losses, non_blocking=True)
        self.torch.cuda.synchronize(self.device)
        # The only D2H/readback of losses; all slots are now independent and complete.
        losses = self.cpu_losses.tolist()
        if not all(math.isfinite(loss) for loss in losses):
            raise RuntimeError("Continuous loop produced a non-finite loss")
        for row, loss in zip(rows, losses):
            row["loss"] = float(loss)
        samples = sum(int(row["batch_size"]) for row in rows)
        barrier()  # Both lanes call the same success-only final WORLD barrier.
        self.torch.cuda.synchronize(self.device)  # Include the barrier's GPU tail.
        stop = time.perf_counter()
        self.finished = True
        self.summary = {
            "status": "complete", "timing_protocol": "continuous_loop_all_steps",
            "elapsed_ms": (stop - self.start_time) * 1e3,
            "final_drain_ms": (stop - drain_start) * 1e3,
            "steps": self.steps, "loss_slots_written": self.written,
            "local_samples": samples, "warmup_steps_excluded": 0,
            "max_pending_events": self.max_pending, "event_depth_limit": 2,
            "final_pending_events": len(self.pending), "losses_finite": True,
            "final_device_drain_complete": True, "final_barrier_complete": True,
            "loss_gpu_bytes": self.steps * 4, "loss_cpu_bytes": self.steps * 4,
            "includes": "loss setup, initial fill, all steps, final training/metadata waits, loss collection, WORLD barrier and device drain",
            "excludes": "model/graph setup, state export, session close and object destruction",
            "aggregation": "sum local_samples / max rank elapsed; never sum per-step host timers",
        }
        return self.summary

    def drain_after_error(self, error):
        # No error-only collective: preserve the original failure even if drain fails.
        if not self.started or self.finished:
            return
        for _, event in self.pending:
            try:
                event.synchronize()
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note("Continuous terminal event drain also failed: " + repr(cleanup_error))
        try:
            self.torch.cuda.synchronize(self.device)
        except BaseException as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note("Continuous device drain also failed: " + repr(cleanup_error))
        self.pending.clear()
