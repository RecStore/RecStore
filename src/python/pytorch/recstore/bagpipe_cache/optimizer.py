"""BagPipe sparse SGD optimizer.

Drop-in replacement for SparseSGD that routes gradient updates through
BagPipeCacheController.update_grads() instead of direct PS pushes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .controller import BagPipeCacheController


class BagPipeSparseSGD:
    """Drop-in replacement for SparseSGD that uses BagPipeCacheController.

    Instead of pushing gradients to the PS via ``update_async``/``wait``,
    this optimizer delegates to ``BagPipeCacheController.update_grads()``
    which applies SGD in-place on the GPU cache and splits updates into
    sync_now (immediate) / sync_later (deferred).
    """

    def __init__(self, params, lr: float, controller: "BagPipeCacheController"):
        self.param_groups = [{"params": params, "lr": float(lr)}]
        self.controller = controller
        self._batch_num = 0

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for mod in group["params"]:
                if hasattr(mod, "reset_trace"):
                    mod.reset_trace()

    def step(self) -> None:
        from python.pytorch.recstore.optimizer import _collect_traces_by_name

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for mod in group["params"]:
                    if not hasattr(mod, "_trace") or not mod._trace:
                        continue

                    traces_by_name = _collect_traces_by_name(mod)
                    for name, entries in traces_by_name.items():
                        if not entries:
                            continue
                        all_ids = torch.cat(
                            [ids for ids, _ in entries], dim=0
                        )
                        all_grads = torch.cat(
                            [grads for _, grads in entries], dim=0
                        )

                        unique_ids, inverse_indices = torch.unique(
                            all_ids, return_inverse=True
                        )
                        summed_grads = torch.zeros(
                            (len(unique_ids), all_grads.size(1)),
                            device=all_grads.device,
                            dtype=all_grads.dtype,
                        )
                        summed_grads.index_add_(0, inverse_indices, all_grads)

                        self.controller.update_grads(
                            name,
                            unique_ids,
                            summed_grads,
                            lr,
                            self._batch_num,
                        )

                    if hasattr(mod, "reset_trace"):
                        mod.reset_trace()

            self._batch_num += 1

    def flush(self) -> None:
        """No-op: BagPipe controller handles writeback at cleanup time."""
        pass
