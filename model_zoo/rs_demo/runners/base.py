from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..config import RunConfig


class BenchmarkRunner(ABC):
    @abstractmethod
    def run(self, repo_root: Path, cfg: RunConfig) -> dict:
        raise NotImplementedError

