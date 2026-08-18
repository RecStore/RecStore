from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..config import RunConfig


def build_torchrec_profiler(
    cfg: RunConfig,
    on_trace_ready: Callable[[object], None] | None = None,
):
    if not cfg.torchrec_profiler:
        return None
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch profiler requires torch to be installed.") from exc

    if not hasattr(torch, "profiler"):
        raise RuntimeError("Torch profiler is unavailable in this torch build.")

    activities: list[object] = []
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    activities.append(torch.profiler.ProfilerActivity.CPU)

    trace_dir = Path(cfg.torchrec_trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = on_trace_ready or torch.profiler.tensorboard_trace_handler(
        str(trace_dir)
    )

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=cfg.torchrec_profiler_warmup,
            warmup=0,
            active=cfg.torchrec_profiler_active,
            repeat=cfg.torchrec_profiler_repeat,
        ),
        on_trace_ready=handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
