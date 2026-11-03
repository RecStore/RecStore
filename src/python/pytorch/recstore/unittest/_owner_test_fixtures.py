"""Shared CPU fixtures for the merged-gradient owner pipeline tests.

Collectives and PS transport are explicit fakes around the real owner pipeline.
"""
import threading

import numpy as np
import torch
import torch.distributed as dist
from unittest import mock

from .. import canonical_ps as planner
from .. import ps_pipeline as pipeline_module


class Features:
    def keys(self): return ['f0', 'f1']


class FakePS:
    def __init__(self):
        self.release, self.waiting = threading.Event(), threading.Event()
        self.release.set()
        self.reads, self.commits, self.closed = [], [], []
        self.failure = None
    def pull(self, table, ids):
        self.reads.append(ids.tolist())
        return ids.float()[:, None].expand(-1, 128).clone()
    def sparse_gradient_commit_wait(self, handle):
        self.waiting.set()
        if not self.release.wait(5): raise TimeoutError('Unreleased fake commit')
        if self.failure is not None: raise self.failure
    def sparse_gradient_commit_close(self, table, **kw):
        self.closed.append(kw)
    def sparse_gradient_commit_async(self, *a, **kw):
        raise AssertionError('Merged owner path used legacy submission')


class OwnerPS(FakePS):
    def sparse_owner_replace_async(self, table, ids, values, status, **kwargs):
        self.commits.append((ids.clone(), values.clone(), kwargs))
        return kwargs['step']

    def sparse_owner_replace_close(self, table, **kwargs):
        self.closed.append(kwargs)


local_batches = [torch.tensor([[1,2],[1,3]]), torch.tensor([[0,2],[1,4],[5,4]]),
                 torch.tensor([[2,1]])]


peer_batches = [torch.tensor([[2,2],[2,3]]), torch.tensor([[0,4],[2,3]]),
                torch.empty((0,2), dtype=torch.int64)]


offsets = np.array([0,100], np.int64)


types = [0,1]


def ticket(batch): return planner.make_ticket(batch.t().numpy(), offsets, types)


class Collective:
    def __init__(self, corrupt=None):
        self.step, self.calls, self.corrupt = None, [], corrupt
    def __call__(self, outputs, value, group):
        if value.ndim == 1:
            self.step = int(value[0])
            remote = ticket(peer_batches[self.step])
            peer = value.clone()
            peer[1], peer[4] = len(remote.keys), remote.batch_size
            if self.corrupt == 'layout': peer[3] += 1
            if self.corrupt == 'batch': peer[4] = 1025
            if self.corrupt == 'width': peer[1] = 1000000
            outputs[0].copy_(value); outputs[1].copy_(peer)
            self.calls.append(('header', self.step, value.numel()))
        else:
            remote = ticket(peer_batches[self.step])
            peer = torch.zeros_like(value)
            peer[:len(remote.keys),0] = torch.from_numpy(remote.keys)
            peer[:len(remote.keys),1] = torch.from_numpy(remote.counts)
            if self.corrupt == 'counts': peer[0,1] += 1
            if self.corrupt == 'keys': peer[:len(remote.keys),0] = peer[:len(remote.keys),0].flip(0)
            outputs[0].copy_(value); outputs[1].copy_(peer)
            self.calls.append(('payload', self.step, value.numel()))


def context(collective):
    return mock.patch.multiple(dist, is_initialized=lambda: True,
        get_world_size=lambda: 2, get_rank=lambda: 0,
        broadcast=lambda *a, **kw: None, new_group=lambda **kw: object(),
        destroy_process_group=lambda group: None, all_gather=collective)
