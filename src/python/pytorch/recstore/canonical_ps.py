"""Count-exchange ticket helpers for the merged-gradient owner pipeline."""
from dataclasses import dataclass

import numpy as np

TW, DP = 0, 1


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
    """Return the local ticket and feature-major inverse."""
    ticket = make_ticket(feature_major_ids, table_offsets, table_types)
    size = ticket.batch_size * ticket.features
    inverse = np.empty(size, dtype=np.int64)
    if size:
        feature_major = ((ticket.positions % ticket.features) * ticket.batch_size
                         + ticket.positions // ticket.features)
        inverse[feature_major] = np.repeat(
            np.arange(len(ticket.keys), dtype=np.int64), ticket.counts)
    return ticket, inverse


@dataclass
class OwnerExchangePlan:
    """Local row count and the native owner routing tensors."""

    local_unique_count: int
    owner: tuple = ()
