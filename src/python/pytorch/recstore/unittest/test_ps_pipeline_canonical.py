"""Canonical metadata and scheduling regression tests on CPU.

Collectives and packed vectors are explicit fakes. The actual planner/pipeline
are exercised; these tests make no claim about GPU reduction arithmetic.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

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
    def sparse_canonical_gradient_commit_async(self, table, records, values, **kw):
        # Observe ownership/routing only. These fake values are never used as SGD.
        self.commits.append((records, values, kw))
        return kw['step']
    def sparse_gradient_commit_wait(self, handle):
        self.waiting.set()
        if not self.release.wait(5): raise TimeoutError('Unreleased fake commit')
        if self.failure is not None: raise self.failure
    def sparse_canonical_gradient_commit_close(self, table, **kw):
        self.closed.append(kw)
    def sparse_gradient_commit_async(self, *a, **kw):
        raise AssertionError('Canonical path used legacy submission')


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


def make_pipeline(ps, split):
    # Deliberately bypass only constructor GPU-capability checks by enabling
    # canonical after constructing the CPU fixture. No claim about CUDA init.
    pipeline = pipeline_module.PSPipeline(None, ps, table_offsets={'f0':0,'f1':100},
        master_table_name='master', embedding_dim=128, device='cpu',
        dependency_split=split, total_steps=3)
    pipeline.canonical = True
    pipeline._canonical_table_types = tuple(types)
    pipeline._canonical_layout = 17
    if pipeline._metadata_worker is None:
        pipeline._metadata_group = object()
        pipeline._metadata_worker = ThreadPoolExecutor(max_workers=1)
    return pipeline


def fake_pack(grad, plan):
    # Counts and descriptors are real. Zero vectors are opaque transport
    # tokens here: this stub does not implement or test reduction arithmetic.
    return plan.records_tensor(), torch.zeros((plan.vector_count,128))


def context(collective):
    return mock.patch.multiple(dist, is_initialized=lambda: True,
        get_world_size=lambda: 2, get_rank=lambda: 0,
        broadcast=lambda *a, **kw: None, new_group=lambda **kw: object(),
        destroy_process_group=lambda group: None, all_gather=collective)


class TestCanonicalPSPipeline(unittest.TestCase):
    def test_each_step_counts_final_empty_peer_and_commit_read_order(self):
        for split in (False, True):
            with self.subTest(split=split):
                ps, collective = FakePS(), Collective()
                with context(collective), mock.patch.object(planner, 'packed_gpu_vectors', fake_pack):
                    pipeline = make_pipeline(ps, split)
                    optimizer = pipeline_module.PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None),pipeline)
                    prepared = {}
                    def prepare(step):
                        row = {}
                        cpu_ticket = pipeline.on_batch_prepared(step, Features(), local_batches[step], row)
                        prepared[step] = (cpu_ticket, row)
                    prepare(0)
                    try:
                        for step in range(3):
                            cpu_ticket,row = prepared[step]
                            pipeline.before_lookup(step, Features(), cpu_ticket, row)
                            expected_ids = (local_batches[step]+torch.from_numpy(offsets)).reshape(-1)
                            self.assertTrue(torch.equal(cpu_ticket[0][pipeline._current_inverse], expected_ids))
                            if step+1 < 3: prepare(step+1)
                            if split and step == 0:
                                pipeline._split_clean.result(timeout=5)
                                self.assertEqual(ps.reads, [[1,102,103],[0,5,104]])
                                self.assertFalse(ps.commits)
                            ps.waiting.clear()
                            if step == 0: ps.release.clear()
                            optimizer.record_pooled_grad(torch.zeros((len(local_batches[step]),2,128)))
                            self.assertNotIn(step,pipeline._canonical_metadata)
                            optimizer.step()
                            self.assertIsNone(pipeline._canonical_records)
                            self.assertEqual(ps.commits[-1][2]['step'],step)
                            self.assertTrue(torch.equal(ps.commits[-1][0][:,0],cpu_ticket[0]))
                            self.assertTrue(ps.waiting.wait(5))
                            if step == 0:
                                self.assertFalse(pipeline._read.done())
                                if split: self.assertEqual(ps.reads,[[1,102,103],[0,5,104]])
                                ps.release.set()
                            own,peer = ticket(local_batches[step]),ticket(peer_batches[step])
                            self.assertEqual(row['ps_canonical_metadata_sent_bytes'],
                                (6+2*max(len(own.keys),len(peer.keys)))*8)
                        self.assertIsNone(pipeline._read, 'Final submission scheduled an extra read')
                        pipeline.shutdown()
                        self.assertEqual([s for kind,s,_ in collective.calls if kind=='header'],[0,1,2])
                        self.assertEqual([s for kind,s,_ in collective.calls if kind=='payload'],[0,1,2])
                        self.assertFalse(pipeline._canonical_metadata or pipeline._metadata)
                        self.assertFalse(ps.closed[-1]['abort'])
                    finally:
                        ps.release.set(); pipeline.shutdown(flush=False)

    def test_bad_metadata_fails_before_any_native_submission(self):
        for bad in ('layout','batch','width','counts','keys'):
            with self.subTest(bad=bad), context(Collective(bad)):
                ps = FakePS(); pipeline = make_pipeline(ps,False)
                try:
                    with self.assertRaises((RuntimeError,ValueError)):
                        pipeline._exchange_canonical_metadata(0,ticket(local_batches[0]))
                    self.assertFalse(ps.commits)
                finally: pipeline.shutdown(flush=False)

    def test_commit_failure_keeps_dirty_unread_and_preserves_primary(self):
        with context(Collective()), mock.patch.object(planner,'packed_gpu_vectors',fake_pack):
            ps = FakePS(); pipeline = make_pipeline(ps,True)
            optimizer = pipeline_module.PSPipelineSGD(SimpleNamespace(reset_trace=lambda:None),pipeline)
            first=pipeline.on_batch_prepared(0,Features(),local_batches[0],{})
            row={}; pipeline.before_lookup(0,Features(),first,row)
            pipeline.on_batch_prepared(1,Features(),local_batches[1],{})
            pipeline._split_clean.result(timeout=5)
            error=ValueError('original native commit failure'); ps.failure=error
            try:
                optimizer.record_pooled_grad(torch.zeros((2,2,128))); optimizer.step()
                with self.assertRaises(ValueError) as caught: pipeline._read.result(timeout=5)
                self.assertIs(caught.exception,error)
                self.assertEqual(ps.reads,[[1,102,103],[0,5,104]])
                def close(*a,**kw): raise OSError('secondary abort error')
                ps.sparse_canonical_gradient_commit_close=close
                with self.assertRaises(ValueError) as caught: pipeline.shutdown()
                self.assertIs(caught.exception,error)
            finally: pipeline.shutdown(flush=False)

