"""Experimental canonical-v3 count planner and GPU reducer; no production imports."""
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np

TW, DP = 0, 1
SERIAL, TREE, RAW = 0, 1, 2
D = 128
DESC_BYTES = 16
HEADER_BYTES = 56


def contributions(c0, c1, rank, table_type):
    """(kind, local occurrence start, count, output vectors), in wire order."""
    if rank not in (0, 1) or table_type not in (TW, DP):
        raise ValueError('Invalid rank or table type')
    if not (0 <= c0 <= 1024 and 0 <= c1 <= 1024):
        raise ValueError('Canonical v3 requires counts in [0,1024]')
    local = (c0, c1)[rank]
    if local == 0:
        return []
    if table_type == DP:
        return [(SERIAL if local < 32 else TREE, 0, local, 1)]
    total = c0 + c1
    if total < 32:
        return [(SERIAL, 0, c0, 1)] if rank == 0 else [
            (SERIAL, 0, c1, 1) if c0 == 0 else (RAW, 0, c1, c1)]
    result = []
    own_start, own_end = (0, c0) if rank == 0 else (c0, total)
    for base in range(0, total, 1024):
        end = min(base + 1024, total)
        if own_start <= base and end <= own_end:
            result.append((TREE, base-own_start, end-base, 1))
            continue
        width = (end-base+31)//32
        for leaf in range(32):
            lo, hi = base+leaf*width, min(base+(leaf+1)*width, end)
            start, stop = max(lo, own_start), min(hi, own_end)
            if start >= stop:
                continue
            crosses = lo < c0 < hi
            kind = RAW if crosses and rank == 1 else SERIAL
            result.append((kind, start-own_start, stop-start,
                           stop-start if kind == RAW else 1))
    return result


@dataclass
class LocalTicket:
    keys: np.ndarray
    counts: np.ndarray
    positions: np.ndarray
    starts: np.ndarray
    table_types: np.ndarray
    batch_size: int
    features: int


def make_ticket(feature_major_ids, table_offsets, table_types):
    """CPU ids[F,B] -> sorted key counts and stable sample-major grad indices."""
    raw = np.asarray(feature_major_ids)
    offsets = np.asarray(table_offsets, dtype=np.int64)
    types = np.asarray(table_types, dtype=np.int64)
    if raw.dtype != np.int64 or raw.ndim != 2:
        raise ValueError('Expected CPU int64 feature-major IDs [F,B]')
    f, b = raw.shape
    if not (1 <= f <= 26 and 0 <= b <= 1024) or offsets.shape != (f,) or types.shape != (f,):
        raise ValueError('Invalid fixed-workload shape/layout')
    if np.any(raw < 0) or np.any(offsets < 0) or np.any((types != TW) & (types != DP)):
        raise ValueError('Invalid logical IDs, offsets or table type')
    if f > 1 and np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError('Distinct feature namespaces must be ordered')
    fused = (raw + offsets[:, None]).reshape(-1)
    if np.any(fused < 0):
        raise ValueError('Logical fused IDs overflow signed int64')
    if b and f > 1 and any(np.any(raw[i] + offsets[i] >= offsets[i+1]) for i in range(f-1)):
        raise ValueError('Feature namespace overlap')
    order = np.argsort(fused, kind='stable')
    sorted_keys = fused[order]
    keys, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    if np.any(counts > 1024):
        raise ValueError('Per-key count exceeds one ID per sample')
    positions = ((order % b)*f + order//b).astype(np.int64) if b else np.empty(0, np.int64)
    feature_index = order[starts]//b if b else np.empty(0, np.int64)
    return LocalTicket(keys, counts.astype(np.int64), positions, starts.astype(np.int64),
                       types[feature_index], b, f)


def make_ticket_with_inverse(feature_major_ids, table_offsets, table_types):
    """Return the canonical ticket and feature-major inverse; native is opt-in."""
    if os.environ.get('RECSTORE_PS_CANONICAL_CPP_TICKET') != '1':
        ticket = make_ticket(feature_major_ids, table_offsets, table_types)
        size = ticket.batch_size * ticket.features
        inverse = np.empty(size, dtype=np.int64)
        if size:
            feature_major = ((ticket.positions % ticket.features) * ticket.batch_size
                             + ticket.positions // ticket.features)
            inverse[feature_major] = np.repeat(
                np.arange(len(ticket.keys), dtype=np.int64), ticket.counts)
        return ticket, inverse
    import torch
    raw = np.asarray(feature_major_ids)
    offsets = np.asarray(table_offsets, dtype=np.int64)
    types = np.asarray(table_types, dtype=np.int64)
    if raw.dtype != np.int64 or raw.ndim != 2:
        raise ValueError('Expected CPU int64 feature-major IDs [F,B]')
    f, b = raw.shape
    if not (1 <= f <= 26 and 0 <= b <= 1024) or offsets.shape != (f,) or types.shape != (f,):
        raise ValueError('Invalid fixed-workload shape/layout')
    op = getattr(torch.ops.recstore_ops, 'canonical_sparse_ticket', None)
    if not callable(op):
        raise RuntimeError('RECSTORE_PS_CANONICAL_CPP_TICKET=1 requires the canonical_sparse_ticket native op')
    # NumPy may mark empty/singleton arrays contiguous despite negative strides.
    arrays = (x.copy(order='C') if any(stride < 0 for stride in x.strides)
              else np.ascontiguousarray(x) for x in (raw, offsets, types))
    keys, counts, positions, starts, key_types, inverse = op(
        *(torch.from_numpy(x) for x in arrays))
    return LocalTicket(keys.numpy(), counts.numpy(), positions.numpy(), starts.numpy(),
                       key_types.numpy(), b, f), inverse.numpy()


@dataclass
class ContributionPlan:
    records: np.ndarray  # CPU int64 [N,5]: key,c0,c1,nvec,type.
    positions: np.ndarray
    tasks: np.ndarray  # int64 [M,6]: kind,pos_start,count,out_start,predivide,postdivide.
    batch_size: int
    features: int
    rank: int
    vector_count: int

    def records_tensor(self):
        import torch
        return torch.from_numpy(self.records.copy())


def _build_plan_cpp(ticket, peer_keys, peer_counts, rank):
    import torch
    if not (0 <= ticket.batch_size <= 1024 and 1 <= ticket.features <= 26):
        raise ValueError('Invalid fixed-workload dimensions')
    positions = np.asarray(ticket.positions)
    if (positions.dtype != np.int64 or positions.ndim != 1 or
            len(positions) != ticket.batch_size * ticket.features or
            np.any(positions < 0) or np.any(positions >= len(positions))):
        raise ValueError('Invalid source gradient positions')
    arrays = [np.asarray(x) for x in
              (ticket.keys, ticket.counts, ticket.starts, ticket.table_types)]
    arrays += [np.asarray(peer_keys, dtype=np.int64), np.asarray(peer_counts, dtype=np.int64)]
    if any(x.dtype != np.int64 or x.ndim != 1 for x in arrays):
        raise ValueError('Expected CPU int64 ticket and peer vectors')
    op = getattr(torch.ops.recstore_ops, 'canonical_sparse_plan', None)
    if not callable(op):
        raise RuntimeError('RECSTORE_PS_CANONICAL_CPP_PLAN=1 requires the canonical_sparse_plan native op')
    records, tasks, vectors = op(
        *(torch.from_numpy(np.ascontiguousarray(x)) for x in arrays), len(positions), rank)
    return ContributionPlan(records.numpy(), positions.copy(), tasks.numpy(),
                            ticket.batch_size, ticket.features, rank, vectors)


def build_plan(ticket, peer_keys, peer_counts, rank):
    """Use actual peer counts; missing keys contribute zero, never a timeout guess."""
    if rank not in (0, 1):
        raise ValueError('world2 rank must be 0 or 1')
    if os.environ.get('RECSTORE_PS_CANONICAL_CPP_PLAN') == '1':
        return _build_plan_cpp(ticket, peer_keys, peer_counts, rank)
    peer_keys = np.asarray(peer_keys, dtype=np.int64)
    peer_counts = np.asarray(peer_counts, dtype=np.int64)
    if peer_keys.ndim != 1 or peer_counts.shape != peer_keys.shape or np.any(peer_keys < 0):
        raise ValueError('Invalid peer key/count payload')
    if np.any(peer_keys[1:] <= peer_keys[:-1]) or np.any(peer_counts < 1) or np.any(peer_counts > 1024):
        raise ValueError('Peer keys must be unique/sorted with valid positive counts')
    match = np.searchsorted(peer_keys, ticket.keys)
    others = np.zeros(len(ticket.keys), np.int64)
    present = match < len(peer_keys)
    present[present] &= peer_keys[match[present]] == ticket.keys[present]
    others[present] = peer_counts[match[present]]
    records, tasks, cursor = [], [], 0
    for i, key in enumerate(ticket.keys):
        local, peer, kind = int(ticket.counts[i]), int(others[i]), int(ticket.table_types[i])
        c0, c1 = (local, peer) if rank == 0 else (peer, local)
        parts = contributions(c0, c1, rank, kind)
        vectors = sum(p[3] for p in parts)
        records.append((int(key), c0, c1, vectors, kind))
        for op, start, count, nvec in parts:
            tasks.append((op, int(ticket.starts[i])+start, count, cursor, int(kind==TW), int(kind==DP)))
            cursor += nvec
    return ContributionPlan(np.asarray(records, np.int64).reshape(-1, 5), ticket.positions.copy(),
                            np.asarray(tasks, np.int64).reshape(-1, 6), ticket.batch_size,
                            ticket.features, rank, cursor)


def chunk_records(records, payload_limit, shard_ids, shards=2):
    """Use caller-verified physical-key routing; never split a key record."""
    if shards < 1 or payload_limit < HEADER_BYTES:
        raise ValueError('Invalid payload/shard limit')
    shard_ids = np.asarray(shard_ids)
    if shard_ids.shape != (len(records),) or np.any(shard_ids < 0) or np.any(shard_ids >= shards):
        raise ValueError("Invalid explicit shard routing")
    result = []
    for shard in range(shards):
        indices = np.flatnonzero(shard_ids == shard)
        chunks, current, size = [], [], HEADER_BYTES
        for i in indices:
            count = DESC_BYTES + int(records[i, 3])*D*4
            if HEADER_BYTES+count > payload_limit:
                raise ValueError('Whole canonical key record exceeds payload limit')
            if current and size+count > payload_limit:
                chunks.append((current, size)); current, size = [], HEADER_BYTES
            current.append(int(i)); size += count
        chunks.append((current, size))  # Includes an empty-rank marker.
        if len(chunks) > 64:
            raise ValueError('Per-rank/shard canonical chunk limit exceeded')
        result.append(chunks)
    return result


_EXTENSION = None


def initialize_gpu_reducer():
    """Compile/load before training barriers; TORCH_EXTENSIONS_DIR selects cache."""
    from torch.utils.cpp_extension import load_inline
    global _EXTENSION
    if _EXTENSION is None:
        source = Path(__file__).with_name('canonical_reduce.cu').read_text()
        _EXTENSION = load_inline(
            name='recstore_canonical_v3_probe',
            cpp_sources='torch::Tensor canonical_reduce(torch::Tensor, torch::Tensor, torch::Tensor, int64_t);',
            cuda_sources=source, functions=['canonical_reduce'],
            extra_cuda_cflags=['--ftz=false', '--fmad=false'], verbose=False)
    return _EXTENSION


class CanonicalGPUPlan:
    """Own immutable per-batch integer plan copies until global commit/drain."""

    def __init__(self, plan, device):
        import torch
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Canonical GPU plan requires a CUDA device')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        positions, tasks = plan.positions, plan.tasks
        if (not isinstance(positions, np.ndarray) or positions.dtype != np.int64
                or positions.shape != (plan.batch_size * plan.features,)
                or not positions.flags.c_contiguous
                or not isinstance(tasks, np.ndarray) or tasks.dtype != np.int64
                or tasks.ndim != 2 or tasks.shape[1] != 6
                or not tasks.flags.c_contiguous):
            raise ValueError('Expected contiguous CPU int64 canonical plan arrays')
        self.plan, self.device = plan, device
        self._arrays = (positions, tasks)
        self._shape = (plan.batch_size, plan.features, plan.vector_count)
        self._host, self._gpu, self._ready = [], [], None
        try:
            for array in self._arrays:
                host = torch.from_numpy(array).pin_memory()
                self._host.append(host)
                self._gpu.append(host.to(device, non_blocking=True))
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(device))
        except BaseException as error:
            try:
                self.drain()
            except BaseException as cleanup:
                error.add_note('Canonical plan construction drain also failed: ' + repr(cleanup))
            raise

    def tensors(self, plan, device):
        import torch
        if (plan is not self.plan or plan.positions is not self._arrays[0]
                or plan.tasks is not self._arrays[1]
                or self._shape != (plan.batch_size, plan.features, plan.vector_count)
                or torch.device(device) != self.device):
            raise ValueError('Canonical GPU plan identity/device mismatch')
        # Arrays are immutable after preparation; no GPU arithmetic changes.
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self._ready)
        for tensor in self._gpu:
            tensor.record_stream(stream)
        return self._gpu

    def drain(self):
        import torch
        torch.cuda.synchronize(self.device)


def packed_gpu_vectors(grad, plan, prepared=None):
    """Returns independent CUDA FP32 values and CPU int64 records on current stream."""
    import torch
    if (grad.device.type != 'cuda' or grad.dtype != torch.float32 or not grad.is_contiguous()
            or tuple(grad.shape) != (plan.batch_size, plan.features, D)):
        raise ValueError('Expected contiguous CUDA FP32 [B,F,128] gradient')
    extension = initialize_gpu_reducer()
    if prepared is None:
        positions = torch.from_numpy(plan.positions).to(grad.device)
        tasks = torch.from_numpy(plan.tasks).to(grad.device)
    else:
        positions, tasks = prepared.tensors(plan, grad.device)
    values = extension.canonical_reduce(grad, positions, tasks, plan.vector_count)
    return plan.records_tensor(), values
