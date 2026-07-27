from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ProfilerConfig:
    """Profiler configuration, decoupled from model_zoo.RunConfig."""

    enabled: bool
    trace_dir: str
    warmup: int = 0
    active: int = 2
    repeat: int = 1


def build_torchrec_profiler(
    cfg: ProfilerConfig,
    on_trace_ready: Callable[[object], None] | None = None,
):
    if not cfg.enabled:
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

    trace_dir = Path(cfg.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = on_trace_ready or torch.profiler.tensorboard_trace_handler(
        str(trace_dir)
    )

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=cfg.warmup,
            warmup=0,
            active=cfg.active,
            repeat=cfg.repeat,
        ),
        on_trace_ready=handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
