"""Isolated v22 candidate: one current-batch owner upload for forward/backward.

Native planning must prove all index bounds, permutations and split agreement.
The caller creates this only after previous GLOBAL commit, launches all WORLD
collectives on one main thread, and retains it until current GLOBAL commit.
"""
import torch
import torch.distributed as dist
from .owner_gpu import GPUOwnerExchange, _LIMIT, _splits


class ForwardGPUOwnerExchange(GPUOwnerExchange):
    def __init__(self, permutation, owner_keys, metadata, send_splits, recv_splits,
                 old_weights, *, forward_send_index, forward_reorder,
                 forward_send_splits, forward_recv_splits, process_group, device, source_refs=()):
        self.forward_send_splits = _splits(forward_send_splits)
        self.forward_recv_splits = _splits(forward_recv_splits)
        for value in (forward_send_index, forward_reorder):
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.int64 or value.ndim != 1
                    or not value.is_contiguous() or value.requires_grad):
                raise ValueError('Forward owner indices must be contiguous CPU int64 vectors')
        if (len(forward_send_index) != sum(self.forward_send_splits)
                or len(forward_reorder) != sum(self.forward_recv_splits)
                or len(forward_reorder) > _LIMIT // 2):
            raise ValueError('Forward owner split/index shape mismatch')
        super().__init__(permutation, owner_keys, metadata, send_splits, recv_splits,
                         old_weights, process_group=process_group, device=device, source_refs=source_refs)
        self._forward_started = self._forward_done = self._failed = False
        self._forward_gpu = []
        try:
            self._forward_host = tuple(x.pin_memory() for x in (forward_send_index, forward_reorder))
            for value in self._forward_host:
                self._forward_gpu.append(value.to(self.device, non_blocking=True))
            self._forward_ready = torch.cuda.Event()
            self._forward_ready.record(torch.cuda.current_stream(self.device))
        except BaseException as original:
            try:
                self.drain()
            except BaseException as cleanup:
                original.add_note('Forward preparation drain also failed: '+repr(cleanup))
            raise

    def forward_rows(self):
        """Return local unique rows in ticket order; no arithmetic or publication."""
        if self._forward_started or self._failed or self._ran:
            raise RuntimeError('Owner forward cannot be reused or retried')
        self._forward_started = True
        try:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self._forward_ready)
            indices, reorder = self._forward_gpu
            weights = self._gpu[2]  # The exact object later consumed by owner_math.
            for value in (weights, indices, reorder):
                value.record_stream(stream)
            self._forward_send = torch.index_select(weights, 0, indices)
            self._forward_received = torch.empty((len(reorder), 128), dtype=torch.float32,
                                                  device=self.device)
            self._forward_work = dist.all_to_all_single(
                self._forward_received, self._forward_send,
                output_split_sizes=self.forward_recv_splits,
                input_split_sizes=self.forward_send_splits, group=self.group, async_op=True)
            self._forward_work.wait()
            self._forward_rows = torch.index_select(self._forward_received, 0, reorder)
            self._forward_done = True
            return self._forward_rows
        except BaseException:
            self._failed = True
            raise

    def run(self, packed_partials, learning_rate):
        if not self._forward_done or self._failed:
            raise RuntimeError('Owner backward requires a successful forward')
        try:
            return super().run(packed_partials, learning_rate)
        except BaseException:
            self._failed = True
            raise
