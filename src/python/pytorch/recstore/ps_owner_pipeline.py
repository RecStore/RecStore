"""Experimental GPU sparse owners with CPU-authoritative, versioned PS storage."""
from __future__ import annotations

import hashlib
import time
from typing import NamedTuple

import numpy as np

import torch
import torch.distributed as dist

from .ps_pipeline import PSPipeline


def _validate_owner_handoff_indices(current, ids, clean, dirty, source):
    # One CPU validation in the I/O worker, before any clean GET or GPU merge.
    for index in (clean, dirty, source):
        if (index.device.type != "cpu" or index.dtype != torch.int64 or index.ndim != 1
                or not index.is_contiguous() or index.requires_grad):
            raise RuntimeError("Invalid owner handoff index tensor")
    count = ids.numel()
    destinations = np.concatenate((clean.numpy(), dirty.numpy()))
    if (destinations.size != count or source.numel() != dirty.numel()
            or (destinations.size and (destinations.min() < 0 or destinations.max() >= count))
            or not np.all(np.bincount(destinations, minlength=count) == 1)):
        raise RuntimeError("Owner handoff destinations must uniquely cover every next row")
    if (source.numel() and (int(source.min()) < 0 or int(source.max()) >= current.numel())):
        raise RuntimeError("Owner handoff source index out of range")
    if not torch.equal(current[source], ids[dirty]):
        raise RuntimeError("Owner handoff source keys disagree with dirty destinations")
    clean_keys = ids[clean].numpy()
    match = np.searchsorted(current.numpy(), clean_keys)
    valid = match < current.numel()
    if np.any(current.numpy()[match[valid]] == clean_keys[valid]):
        raise RuntimeError("Owner handoff clean set contains a current write")


class _OwnerCleanRows(NamedTuple):
    session: int
    step: int
    rank: int
    current_keys: torch.Tensor
    next_keys: torch.Tensor
    clean_pos: torch.Tensor
    dirty_pos: torch.Tensor
    candidate_pos: torch.Tensor
    clean_values: torch.Tensor
    stats: dict


class _OwnerHandoffRows(NamedTuple):
    clean: _OwnerCleanRows
    candidate: torch.Tensor
    ready: object

    @property
    def shape(self):
        return (self.clean.next_keys.numel(), 128)


class GPUOwnerPipeline(PSPipeline):
    """Route canonical partials to GPUs; publish exact new rows through PS v4.

    Rank r owns physical city_hash shard r. The initialized namespace is
    immutable except for this trainer session. Old owner rows are read only
    after the previous global commit; GPU rows live for one current batch.
    """

    def __init__(self, embedding_module, kv_client, *, owner_table_end, unified_forward=False,
                 canonical_gpu_plan=False, candidate_handoff=False, verify_handoff=False, **kwargs):
        if not isinstance(verify_handoff, bool) or (verify_handoff and not candidate_handoff):
            raise ValueError("Handoff verification requires acknowledged candidate handoff")
        self.verify_handoff = verify_handoff
        self._verified_owner_weights = None
        if not isinstance(candidate_handoff, bool) or (candidate_handoff and not unified_forward):
            raise ValueError("Acknowledged candidate handoff requires unified owner forward")
        self.candidate_handoff = candidate_handoff
        self._owner_candidate = None
        if not isinstance(canonical_gpu_plan, bool):
            raise ValueError("canonical_gpu_plan must be a bool")
        self.canonical_gpu_plan = canonical_gpu_plan
        self._canonical_gpu_staging = None
        if not isinstance(unified_forward, bool):
            raise ValueError("unified_forward must be a bool")
        self.unified_forward = unified_forward
        self._pending_owner_step = None
        self._owner_failed = False
        if kwargs.get("canonical_table_types") is None:
            raise ValueError("GPU owner updates require the exact canonical table plan")
        if not kwargs.get("dependency_split"):
            raise ValueError("GPU owner updates currently require dependency splitting")
        plan_op = "canonical_owner_forward_plan" if unified_forward else "canonical_owner_plan"
        for name in (plan_op, "sparse_owner_replace_async",
                     "sparse_owner_replace_close"):
            if not callable(getattr(kv_client.ops, name, None)):
                raise RuntimeError("GPU owner path requires native op " + name)
        offsets = list(kwargs["table_offsets"].values())
        self._owner_boundaries = torch.tensor(offsets + [owner_table_end], dtype=torch.int64)
        self._owner_types = torch.tensor(kwargs["canonical_table_types"], dtype=torch.int64)
        _, self._owner_tag = kv_client.get_embedding_table_namespace(kwargs["master_table_name"])
        self._owner_plan = self._owner_read = self._owner_exchange = None
        from .owner_gpu import initialize_owner_math
        initialize_owner_math(torch.device(kwargs["device"]))
        super().__init__(embedding_module, kv_client, **kwargs)
        # Peer vector splits must agree on the real encoded-key ownership too.
        identity = (self._canonical_layout.to_bytes(8, "little")
                    + self._owner_tag.to_bytes(8, "little")
                    + self._owner_boundaries.numpy().tobytes()
                    + b"owner-v4/world2/city_hash/rank-equals-shard"
                    + (b"/unified-forward-v1" if unified_forward else b"")
                    + (b"/canonical-gpu-plan-v1" if canonical_gpu_plan else b"")
                    + (b"/acknowledged-handoff-v1" if candidate_handoff else b"")
                    + (b"/verify-handoff-v1" if verify_handoff else b""))
        self._canonical_layout = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") & (2**63 - 1)
        if dist.get_backend() != "nccl":
            self.shutdown(flush=False)
            raise ValueError("GPU owner path requires NCCL WORLD with one rank per shard")

    def _augment_canonical_plan(self, plan, peers, sizes):
        columns = [peer[:size, column].contiguous()
                   for peer, size in zip(peers, sizes) for column in (0, 1)]
        operation = (self.kv_client.ops.canonical_owner_forward_plan if self.unified_forward
                     else self.kv_client.ops.canonical_owner_plan)
        owner = operation(
            *columns, self._owner_boundaries, self._owner_types, self._owner_tag, self._rank)
        if len(owner) != (9 if self.unified_forward else 5):
            raise RuntimeError("Unexpected owner plan schema")
        if owner[0].numel() != plan.vector_count:
            raise RuntimeError("Owner routing disagrees with canonical packed vector count")
        plan.owner = owner

    def on_batch_prepared(self, step, sparse_features, sparse_batch, row):
        if (self.unified_forward or self.canonical_gpu_plan) and self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        return super().on_batch_prepared(step, sparse_features, sparse_batch, row)

    def _queue_prepared_read(self, step, ids):
        if not self.unified_forward:
            return super()._queue_prepared_read(step, ids)
        if self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        if step == 0:
            self._read = self._worker.submit(
                self._pull_owned_step, step, self._canonical_metadata[step], self._commit)
        else:
            if self._pending_owner_step is not None:
                raise RuntimeError("Unified owner prepares only one next step")
            if self.candidate_handoff:
                previous_read = self._read
                if previous_read is None:
                    previous_read = self._consumed_read
                if previous_read is None:
                    raise RuntimeError("Current complete owner read was not published")
                self._split_clean = self._worker.submit(
                    self._read_owner_clean, step, previous_read,
                    self._canonical_metadata[step - 1], self._canonical_metadata[step])
            self._pending_owner_step = step
            # Only canonical_metadata is consumed by backward in this mode.
            self._metadata.pop(step - 1, None)

    def _require_next_read(self):
        if not self.unified_forward:
            return super()._require_next_read()
        if self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        if (self._submitted + 1 < self._total_steps
                and self._pending_owner_step != self._submitted + 1):
            raise RuntimeError("Prepare next owner metadata before current submission")
        if (self.candidate_handoff and self._submitted + 1 < self._total_steps
                and self._split_clean is None):
            raise RuntimeError("Prepare next owner clean read before current submission")

    def _queue_next_read(self, commit):
        if not self.unified_forward:
            return super()._queue_next_read(commit)
        if self._pending_owner_step is not None:
            step = self._pending_owner_step
            if step != self._submitted:
                raise RuntimeError("Owner read step differs from accepted submission")
            # The commit-wait task is already queued on this same single worker.
            if self.candidate_handoff:
                if self._owner_candidate is None or self._split_clean is None:
                    raise RuntimeError("Owner handoff lacks candidate or clean read")
                self._read = self._worker.submit(
                    self._finish_owner_handoff, step, self._split_clean, commit, self._owner_candidate)
                self._split_clean = self._owner_candidate = None
            else:
                self._read = self._worker.submit(
                    self._pull_owned_step, step, self._canonical_metadata[step], commit)
            self._pending_owner_step = None

    def _read_owner_clean(self, step, previous_read, current_metadata, next_metadata):
        # Only earlier read and independent metadata futures are awaited here.
        previous_read.result()
        start = time.perf_counter()
        current_result = current_metadata.result()
        current = current_result[2].owner[1]
        ids = next_metadata.result()[2].owner[1]
        plan_wait = (time.perf_counter() - start) * 1e3
        start = time.perf_counter()
        clean_pos, dirty_pos, matches = self._partition_clean_rows(ids, current)
        candidate_pos = matches[dirty_pos].contiguous()
        _validate_owner_handoff_indices(current, ids, clean_pos, dirty_pos, candidate_pos)
        plan_ms = (time.perf_counter() - start) * 1e3
        values, _, elapsed = self._pull(ids[clean_pos], None)
        stats = dict(current_result[1])
        stats.update(ps_owner_plan_wait_ms=plan_wait, ps_owner_handoff_plan_ms=plan_ms,
                     ps_owner_current_rows=current.numel(), ps_pipeline_metadata_wait_ms=plan_wait,
                     ps_pipeline_split_plan_ms=plan_ms, ps_pipeline_clean_pull_ms=elapsed,
                     ps_pipeline_clean_rows=clean_pos.numel(), ps_pipeline_dirty_rows=dirty_pos.numel(),
                     ps_pipeline_clean_read_bytes=values.numel() * values.element_size())
        return _OwnerCleanRows(self._session, step, self._rank, current, ids,
                               clean_pos, dirty_pos, candidate_pos, values, stats)

    def _finish_owner_handoff(self, step, clean, commit, candidate):
        waited = commit.result()  # Successful GLOBAL ACK, including empty owners.
        parts = clean.result()
        session, previous_step, rank, keys, values, ready = candidate
        if ((session, previous_step + 1, rank) != (self._session, step, self._rank)
                or (parts.session, parts.step, parts.rank) != (session, step, rank)
                or keys is not parts.current_keys
                or tuple(values.shape) != (keys.numel(), 128)):
            raise RuntimeError("Owner handoff provenance/row count mismatch")
        stats = dict(parts.stats)
        stats.update(ps_owner_read_step=step, ps_owner_rows=parts.next_keys.numel(),
                     ps_owner_bootstrap_rows=0,
                     ps_owner_old_read_bytes=parts.clean_values.numel() * 4,
                     ps_owner_pull_ms=stats["ps_pipeline_clean_pull_ms"],
                     ps_owner_handoff_rows=parts.dirty_pos.numel(),
                     ps_owner_handoff_bytes=parts.dirty_pos.numel() * 128 * 4,
                     ps_owner_dirty_get_bytes=0,
                     ps_pipeline_dirty_pull_ms=0.0, ps_pipeline_dirty_read_bytes=0,
                     ps_pipeline_split_merge_ms=0.0)
        # No prior Future, exchange, or source holder appears in this result.
        return _OwnerHandoffRows(parts, values, ready), waited, stats["ps_owner_pull_ms"], stats

    def _merge_owner_handoff(self, step, values, keys):
        parts = values.clean
        if ((parts.session, parts.step, parts.rank) != (self._session, step, self._rank)
                or keys is not parts.next_keys
                or parts.clean_pos.numel() + parts.dirty_pos.numel() != keys.numel()
                or parts.candidate_pos.numel() != parts.dirty_pos.numel()
                or tuple(parts.clean_values.shape) != (parts.clean_pos.numel(), 128)
                or tuple(values.candidate.shape) != (parts.current_keys.numel(), 128)
                or parts.clean_values.device.type != "cpu"
                or parts.clean_values.dtype != torch.float32
                or values.candidate.device != self.device
                or values.candidate.dtype != torch.float32
                or values.candidate.requires_grad or not values.candidate.is_contiguous()):
            raise RuntimeError("Invalid acknowledged owner handoff")
        for index, bound in ((parts.clean_pos, keys.numel()), (parts.dirty_pos, keys.numel()),
                             (parts.candidate_pos, parts.current_keys.numel())):
            if (index.device.type != "cpu" or index.dtype != torch.int64 or index.ndim != 1
                    or not index.is_contiguous() or index.requires_grad
                    or (index.numel() and (int(index.min()) < 0 or int(index.max()) >= bound))):
                raise RuntimeError("Invalid owner handoff index")
        refs = [parts.clean_values, values.candidate]
        try:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(values.ready)
            values.candidate.record_stream(stream)
            output = torch.empty((keys.numel(), 128), dtype=torch.float32, device=self.device)
            if parts.clean_pos.numel():
                clean_host_index = parts.clean_pos.pin_memory()
                refs.append(clean_host_index)
                clean_index = clean_host_index.to(self.device, non_blocking=True)
                refs.append(clean_index)
                transferred = parts.clean_values.to(self.device, non_blocking=True)
                refs.append(transferred)
                output.index_copy_(0, clean_index, transferred)
            if parts.dirty_pos.numel():
                for index in (parts.dirty_pos, parts.candidate_pos):
                    host = index.pin_memory()
                    refs.append(host)
                    refs.append(host.to(self.device, non_blocking=True))
                dirty_index, source_index = refs[-3], refs[-1]
                selected = values.candidate.index_select(0, source_index)
                refs.append(selected)
                output.index_copy_(0, dirty_index, selected)
            output.record_stream(stream)
            return output, tuple(refs)  # Flat tensors only; no previous helper chain.
        except BaseException as error:
            try:
                torch.cuda.synchronize(self.device)
            except BaseException as cleanup:
                error.add_note("Owner handoff merge drain also failed: " + repr(cleanup))
            raise

    def _pull_owned_step(self, step, metadata, commit):
        waited = commit.result() if commit is not None else 0.0
        start = time.perf_counter()
        plan = metadata.result()[2].owner
        plan_wait_ms = (time.perf_counter() - start) * 1e3
        values, _, pull_ms = self._pull(plan[1], None)
        return values, waited, pull_ms, {
            "ps_owner_read_step": step,
            "ps_owner_plan_wait_ms": plan_wait_ms,
            "ps_owner_rows": plan[1].numel(),
            "ps_owner_old_read_bytes": values.numel() * 4,
            "ps_owner_pull_ms": pull_ms,
            "ps_owner_bootstrap_rows": values.shape[0] if self.candidate_handoff else 0,
        }

    def _stage_rows(self, step, ticket, values, row_count, row):
        if not self.unified_forward:
            return super()._stage_rows(step, ticket, values, row_count, row)
        from .owner_forward_gpu import ForwardGPUOwnerExchange
        plan = self._canonical_metadata[step].result()[2].owner
        permutation, keys, metadata, send, recv, index, fsend, frecv, reorder = plan
        if reorder.numel() != ticket[0].numel():
            raise RuntimeError("Owner forward reorder differs from local ticket")
        # Completed owned read proves prior GLOBAL ACK before old staging drops.
        self._owner_exchange = None
        self._owner_plan = plan
        start = time.perf_counter()
        source_refs = ()
        try:
            if isinstance(values, _OwnerHandoffRows):
                if not self.candidate_handoff:
                    raise RuntimeError("Unexpected owner handoff in legacy mode")
                merge_start = time.perf_counter()
                values, source_refs = self._merge_owner_handoff(step, values, keys)
                row["ps_owner_handoff_merge_enqueue_ms"] = (time.perf_counter() - merge_start) * 1e3
                row["ps_owner_handoff_source_refs"] = len(source_refs)
                row["ps_owner_handoff_source_bytes"] = sum(x.numel() * x.element_size() for x in source_refs)
                row["ps_owner_handoff_source_cpu_bytes"] = sum(
                    x.numel() * x.element_size() for x in source_refs if x.device.type == "cpu")
                row["ps_owner_handoff_source_cuda_bytes"] = sum(
                    x.numel() * x.element_size() for x in source_refs if x.device.type == "cuda")
            self._owner_exchange = ForwardGPUOwnerExchange(
                permutation, keys, metadata, send, recv, values,
                forward_send_index=index, forward_reorder=reorder,
                forward_send_splits=fsend, forward_recv_splits=frecv,
                process_group=dist.group.WORLD, device=self.device, source_refs=source_refs)
            if self.verify_handoff:
                # Diagnostic only: extra full PS GET/synchronization, never a performance lane.
                verify_start = time.perf_counter()
                if self._submitted != step or (step > 0 and
                        (self._commit is None or not self._commit.done())):
                    raise RuntimeError("Owner verification lacks a completed global commit")
                if step > 0:
                    self._commit.result()
                expected, _, _ = self._pull(keys, None)
                weights = self._owner_exchange._gpu[2]
                actual = weights.detach().cpu()
                if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
                    raise RuntimeError("Owner handoff differs from authoritative PS FP32 bits")
                self._verified_owner_weights = weights
                row.update(ps_owner_handoff_verified_rows=keys.numel(),
                           ps_owner_handoff_verify_passed=1,
                           ps_owner_handoff_verify_ms=(time.perf_counter() - verify_start) * 1e3)
            self._current_rows = self._owner_exchange.forward_rows()
        except BaseException as error:
            if self.candidate_handoff:
                try:
                    torch.cuda.synchronize(self.device)
                except BaseException as cleanup:
                    error.add_note("Owner handoff staging drain also failed: " + repr(cleanup))
            raise
        row.update(ps_owner_forward_enqueue_ms=(time.perf_counter() - start) * 1e3,
                   ps_owner_local_lookup_rows=ticket[0].numel(),
                   ps_owner_forward_peer_send_bytes=int(fsend[1-self._rank]) * 128 * 4,
                   ps_owner_forward_peer_recv_bytes=int(frecv[1-self._rank]) * 128 * 4)

    def before_lookup(self, step, sparse_features, ticket, row):
        for name in ("bootstrap_rows", "handoff_rows", "handoff_bytes", "dirty_get_bytes",
                     "current_rows", "handoff_plan_ms", "handoff_merge_enqueue_ms",
                     "handoff_source_refs", "handoff_source_bytes",
                     "handoff_source_cpu_bytes", "handoff_source_cuda_bytes",
                     "handoff_verified_rows", "handoff_verify_passed",
                     "handoff_same_tensor_passed", "handoff_verify_ms"):
            row["ps_owner_" + name] = 0
        if (self.unified_forward or self.canonical_gpu_plan) and self._owner_failed:
            raise RuntimeError("GPU owner session has failed")
        if self.unified_forward:
            try:
                super().before_lookup(step, sparse_features, ticket, row)
                self._prepare_canonical_gpu_plan(step, row)
                if self.candidate_handoff:
                    row["ps_pipeline_read_bytes"] = row["ps_owner_old_read_bytes"]
                return
            except BaseException:
                self._owner_failed = True
                raise
        try:
            super().before_lookup(step, sparse_features, ticket, row)
            # The completed ordinary read proves the prior GLOBAL commit. Retain the
            # same commit future explicitly for the owner read as a second guard.
            start = time.perf_counter()
            plan = self._canonical_metadata[step].result()[2]
            self._owner_plan = plan.owner
            self._owner_exchange = None
            self._owner_read = self._worker.submit(self._pull, self._owner_plan[1], self._commit)
            row["ps_owner_plan_wait_ms"] = (time.perf_counter() - start) * 1e3
            row["ps_owner_rows"] = self._owner_plan[1].numel()
            row["ps_owner_old_read_bytes"] = self._owner_plan[1].numel() * 128 * 4
            self._prepare_canonical_gpu_plan(step, row)
        except BaseException:
            if self.canonical_gpu_plan:
                self._owner_failed = True
            raise

    def _prepare_canonical_gpu_plan(self, step, row):
        row["ps_canonical_gpu_plan_stage_ms"] = 0.0
        row["ps_canonical_gpu_plan_bytes"] = 0
        if not self.canonical_gpu_plan:
            return
        try:
            from .canonical_ps import CanonicalGPUPlan
            plan = self._canonical_metadata[step].result()[2]
            start = time.perf_counter()
            # before_lookup completed the previous GLOBAL ACK/read.
            # Retain this plan through current native completion.
            self._canonical_gpu_staging = CanonicalGPUPlan(plan, self.device)
            row["ps_canonical_gpu_plan_stage_ms"] = (time.perf_counter() - start) * 1e3
            row["ps_canonical_gpu_plan_bytes"] = plan.positions.nbytes + plan.tasks.nbytes
        except BaseException:
            self._owner_failed = True
            raise

    def _pack_canonical_vectors(self, grad, plan):
        if not self.canonical_gpu_plan:
            return super()._pack_canonical_vectors(grad, plan)
        try:
            from .canonical_ps import packed_gpu_vectors
            if self._owner_failed or self._canonical_gpu_staging is None:
                raise RuntimeError("Canonical GPU plan was not prepared")
            return packed_gpu_vectors(grad, plan, self._canonical_gpu_staging)
        except BaseException:
            self._owner_failed = True
            raise

    def _submit_canonical(self, records, grads):
        if not (self.unified_forward or self.canonical_gpu_plan):
            return self._submit_owner_canonical(records, grads)
        if self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        try:
            return self._submit_owner_canonical(records, grads)
        except BaseException:
            self._owner_failed = True
            raise

    def _submit_owner_canonical(self, records, grads):
        start = time.perf_counter()
        if self.unified_forward:
            if self._owner_exchange is None or self._owner_plan is None:
                raise RuntimeError("Owner submission lacks completed forward staging")
            permutation, keys, metadata, send, recv = self._owner_plan[:5]
            pull_ms = self._current_metrics["ps_owner_pull_ms"]
            read_done = start
        else:
            if self._owner_read is None or self._owner_plan is None:
                raise RuntimeError("GPU owner submission lacks a committed old-weight read")
            from .owner_gpu import GPUOwnerExchange
            weights, _, pull_ms = self._owner_read.result()
            read_done = time.perf_counter()
            permutation, keys, metadata, send, recv = self._owner_plan
            self._owner_exchange = GPUOwnerExchange(
                permutation, keys, metadata, send.tolist(), recv.tolist(), weights,
                process_group=dist.group.WORLD, device=self.device)
        if self.verify_handoff:
            if self._owner_exchange._gpu[2] is not self._verified_owner_weights:
                raise RuntimeError("Owner backward does not use verified forward weights")
            self._current_metrics["ps_owner_handoff_same_tensor_passed"] = 1
        candidates, status = self._owner_exchange.run(grads, self.lr)
        if self.candidate_handoff and self._submitted + 1 < self._total_steps:
            if self._owner_candidate is not None:
                raise RuntimeError("Owner candidate already awaits handoff")
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            self._owner_candidate = (self._session, self._submitted, self._rank,
                                     keys, candidates, ready)
        enqueued = time.perf_counter()
        handle = self.kv_client.sparse_owner_replace_async(
            self.master_table_name, keys, candidates, status, session=self._session,
            step=self._submitted, rank=self._rank, world=self._world, lr=self.lr)
        self._current_metrics.update(
            ps_owner_read_wait_ms=(read_done - start) * 1e3,
            ps_owner_pull_ms=pull_ms,
            ps_owner_exchange_enqueue_ms=(enqueued - read_done) * 1e3,
            ps_owner_commit_submit_ms=(time.perf_counter() - enqueued) * 1e3,
            ps_owner_peer_send_bytes=int(send[1 - self._rank]) * 128 * 4,
            ps_owner_peer_recv_bytes=int(recv[1 - self._rank]) * 128 * 4,
            ps_owner_candidate_bytes=candidates.numel() * 4)
        return handle

    def _close_canonical(self, abort):
        self.kv_client.sparse_owner_replace_close(
            self.master_table_name, session=self._session, step=self._submitted - 1,
            rank=self._rank, world=self._world, lr=self.lr, abort=abort)

    def shutdown(self, *, flush=True):
        error = None
        try:
            if self.candidate_handoff and flush:
                # Observe a failed commit/read before secondary cleanup errors.
                self._wait_pending()
                if self._read is not None:
                    self._read.result()
        except BaseException as exc:
            error = exc
        try:
            if self.candidate_handoff:
                # Read futures may own merge sources: drain before base clears them.
                torch.cuda.synchronize(self.device)
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note("Owner handoff cleanup also failed: " + repr(exc))
        try:
            if self._canonical_gpu_staging is not None:
                self._canonical_gpu_staging.drain()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note("Canonical staging cleanup also failed: " + repr(exc))
        try:
            super().shutdown(flush=flush and error is None)
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note("PS cleanup also failed: " + repr(exc))
        try:
            # Also drain a partially enqueued exchange after a local exception.
            # Successful native completion already covers candidate/status D2H.
            if self._owner_exchange is not None:
                self._owner_exchange.drain()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note("GPU owner cleanup also failed: " + repr(exc))
        finally:
            self._owner_exchange = self._owner_read = self._owner_plan = None
            self._pending_owner_step = None
            self._canonical_gpu_staging = None
            self._owner_candidate = None
            self._verified_owner_weights = None
        if error is not None:
            raise error
