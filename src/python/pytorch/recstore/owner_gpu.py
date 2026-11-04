"""World2 merged-owner exchange with one batch of GPU staging."""
import math
from numbers import Integral, Real

import torch
import torch.distributed as dist

_LIMIT = 2 * 26 * 1024


def _cuda_device(device):
    if isinstance(device, bool):
        raise ValueError('Owner exchange requires a CUDA device')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Owner exchange requires a CUDA device')
    return torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)


def _splits(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.dtype != torch.int64 or tuple(value.shape) != (2,):
            raise ValueError('Owner splits must be CPU int64[2]')
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
            isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in value):
        raise ValueError('Owner splits require two nonnegative integer vector counts')
    if sum(value) > _LIMIT:
        raise ValueError('Owner vector limit exceeded')
    return [int(x) for x in value]


class GPUOwnerExchange:
    """Single use, native-validated plan; caller retains this until global ACK.

    Group ranks must equal physical owner ranks. Packed merged rows must be
    ready on the calling stream. No arithmetic reduction occurs in the
    collective; the single validated merged-row kernel applies it afterward.
    """
    def __init__(self, permutation, owner_keys, metadata, send_splits, recv_splits,
                 old_weights, *, process_group, device, source_refs=()):
        self.device = _cuda_device(device)
        self._check_math_ready()
        if (process_group is None or not dist.is_initialized() or
                dist.get_backend(process_group) != 'nccl' or
                dist.get_world_size(process_group) != 2 or dist.get_world_size() != 2 or
                dist.get_rank(process_group) != dist.get_rank()):
            raise ValueError('Owner exchange requires explicit world2 NCCL group with identity ranks')
        self.send_splits, self.recv_splits = _splits(send_splits), _splits(recv_splits)
        for tensor, dtype, ndim in ((permutation, torch.int64, 1), (owner_keys, torch.int64, 1),
                                   (metadata, torch.int64, 2)):
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or
                    tensor.dtype != dtype or tensor.ndim != ndim or not tensor.is_contiguous() or
                    tensor.requires_grad):
                raise ValueError('Invalid CPU owner plan/weight tensor')
        if (not isinstance(old_weights, torch.Tensor) or
                old_weights.dtype != torch.float32 or old_weights.ndim != 2 or
                not old_weights.is_contiguous() or old_weights.requires_grad or
                (old_weights.device.type != "cpu" and old_weights.device != self.device) or
                not isinstance(source_refs, tuple) or
                any(not isinstance(x, torch.Tensor) for x in source_refs)):
            raise ValueError("Invalid owner weights or flat tensor source references")
        self._source_refs = source_refs
        if (len(owner_keys) > _LIMIT or tuple(metadata.shape) != (len(owner_keys), 8) or
                tuple(old_weights.shape) != (len(owner_keys), 128) or len(permutation) != sum(self.send_splits)):
            raise ValueError('Owner plan/weight shape mismatch')
        # Native planning proves key/count pairing, offsets and permutation.
        self.owner_keys, self.group = owner_keys, process_group
        self._ran, self._work = False, None
        self._gpu = []
        try:
            # Supplied CUDA weights are already ready on the calling stream.
            cpu = (metadata, permutation) if old_weights.is_cuda else (metadata, permutation, old_weights)
            self._host = tuple(x.pin_memory() for x in cpu)
            for tensor in self._host:
                self._gpu.append(tensor.to(self.device, non_blocking=True))
            if old_weights.is_cuda:
                old_weights.record_stream(torch.cuda.current_stream(self.device))
                self._gpu.append(old_weights)
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(self.device))
        except BaseException as original:
            try:
                self.drain()
            except BaseException as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note('Owner preparation drain also failed: '+repr(cleanup))
            raise

    def _check_math_ready(self):
        from . import owner_merged_gpu
        if (owner_merged_gpu._EXTENSION is None
                or self.device.index not in owner_merged_gpu._WARMED_DEVICES):
            raise RuntimeError('Call initialize_owner_merged_math before training')

    def _owner_math(self, meta, v0, v1, weights, learning_rate):
        from .owner_merged_gpu import _EXTENSION
        return _EXTENSION.owner_merged_math(
            meta, v0, v1, weights, float(learning_rate))

    def run(self, packed_rows, learning_rate):
        if self._ran:
            raise RuntimeError('An owner exchange cannot be retried or reused')
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, Real) or
                not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError('Owner arithmetic requires a finite positive learning rate')
        if (not isinstance(packed_rows, torch.Tensor) or packed_rows.device != self.device or
                packed_rows.dtype != torch.float32 or not packed_rows.is_contiguous() or
                tuple(packed_rows.shape) != (sum(self.send_splits), 128) or packed_rows.requires_grad):
            raise ValueError('Expected current-stream contiguous CUDA FP32 merged owner rows')
        self._ran = True
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self._ready)
        for tensor in (*self._gpu, packed_rows):
            tensor.record_stream(stream)
        self._packed = packed_rows
        meta, permutation, weights = self._gpu
        self._send = torch.index_select(packed_rows, 0, permutation)
        self._received = torch.empty((sum(self.recv_splits), 128), dtype=torch.float32, device=self.device)
        self._work = dist.all_to_all_single(self._received, self._send,
            output_split_sizes=self.recv_splits, input_split_sizes=self.send_splits,
            group=self.group, async_op=True)
        self._work.wait()  # Establish NCCL completion on this CUDA stream.
        v0, v1 = self._received.split(self.recv_splits, dim=0)
        self._result = self._owner_math(meta, v0, v1, weights, learning_rate)
        return self._result

    def drain(self):
        """Error cleanup only; successful global ACK already covers producer work."""
        torch.cuda.synchronize(self.device)
