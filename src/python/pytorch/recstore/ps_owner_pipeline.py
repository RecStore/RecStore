"""Experimental GPU sparse owners with CPU-authoritative, versioned PS storage."""
from __future__ import annotations

import hashlib
import time
from typing import NamedTuple
from concurrent.futures import Future

import numpy as np

import torch
import torch.distributed as dist

from .ps_pipeline import PSPipeline


def owner_options_from_env(environ):
    """Enable the single validated merged-owner profile."""
    return {} if environ.get("RECSTORE_PS_OWNER_GPU") == "1" else None


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


class _CPUPreparation(NamedTuple):
    step: int
    source: torch.Tensor
    future: Future


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
    """Route merged-gradient rows to GPUs; publish exact new rows through PS v4.

    Rank r owns physical city_hash shard r. The initialized namespace is
    immutable except for this trainer session. Old owner rows are read only
    after the previous global commit unless bounded compute-ahead is enabled.
    Compute-ahead retains one predecessor while only the CPU-retired version is
    externally authoritative; no external readers/writers/checkpoints are allowed.
    """

    owner_merged_grad = True
    owner_updates = True
    metadata_lookahead = 4

    def __init__(self, embedding_module, kv_client, *, owner_table_end, **kwargs):
        self._metadata_enqueued = self._metadata_queue_hwm = 0
        self._metadata_stopping = False
        self._metadata_prepare_error = None
        self._retired_step = self._accepted_step = self._candidate_step = -1
        self._commit_step = -1
        self._compute_read_step = None
        self._retiring_owner = self._compute_error = None
        self._owner_candidate = None
        self._pending_owner_step = None
        self._owner_failed = False
        owner_table_types = kwargs.pop("owner_table_types", None)
        self._owner_table_types = tuple(
            owner_table_types or [0] * len(kwargs["table_offsets"]))
        self._owner_metadata = {}
        if not kwargs.get("dependency_split"):
            raise ValueError("GPU owner updates currently require dependency splitting")
        world = dist.get_world_size() if dist.is_initialized() else 1
        device = torch.device(kwargs["device"])
        total_steps = kwargs.get("total_steps")
        if (world != 2 or kwargs["embedding_dim"] != 128 or device.type != "cuda"
                or len(self._owner_table_types) != len(kwargs["table_offsets"])
                or any(kind not in (0, 1) for kind in self._owner_table_types)
                or not isinstance(total_steps, int) or isinstance(total_steps, bool)
                or total_steps < 1):
            raise ValueError("Owner PS requires world2, CUDA D128 and an explicit table layout")
        for name in ("owner_merged_forward_plan", "sparse_owner_replace_async",
                     "sparse_owner_replace_close"):
            if not callable(getattr(kv_client.ops, name, None)):
                raise RuntimeError("GPU owner path requires native op " + name)
        offsets = list(kwargs["table_offsets"].values())
        self._owner_boundaries = torch.tensor(offsets + [owner_table_end], dtype=torch.int64)
        self._owner_types = torch.tensor(self._owner_table_types, dtype=torch.int64)
        _, self._owner_tag = kv_client.get_embedding_table_namespace(kwargs["master_table_name"])
        self._owner_plan = self._owner_exchange = None
        from .owner_merged_gpu import initialize_owner_merged_math
        initialize_owner_merged_math(torch.device(kwargs["device"]))
        super().__init__(embedding_module, kv_client, **kwargs)
        layout = (self._offsets.numpy().tobytes() + bytes(self._owner_table_types)
                  + "\0".join(self._keys).encode("utf-8"))
        self._layout_digest = int.from_bytes(
            hashlib.sha256(layout).digest()[:8], "little") & (2**63 - 1)
        # Peer vector splits must agree on the real encoded-key ownership too.
        identity = (self._layout_digest.to_bytes(8, "little")
                    + self._owner_tag.to_bytes(8, "little")
                    + self._owner_boundaries.numpy().tobytes()
                    + b"owner-v4/world2/city_hash/rank-equals-shard"
                    + b"/unified-forward-v1/acknowledged-handoff-v1"
                    + b"/compute-ahead-v1/cpu-metadata-v1/\x04/merged-grad-v1")
        self._layout_digest = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") & (2**63 - 1)
        if dist.get_backend() != "nccl":
            self.shutdown(flush=False)
            raise ValueError("GPU owner path requires NCCL WORLD with one rank per shard")

    @property
    def metadata_queue_high_watermark(self):
        return self._metadata_queue_hwm

    def finish_training(self):
        super().finish_training()
        if self._prepared != self._total_steps or self._metadata_enqueued != self._total_steps:
            raise RuntimeError("Metadata lookahead did not consume the exact tail")
        self._metadata_stopping = True
        self._metadata_worker.shutdown(wait=True, cancel_futures=False)

    def fail(self, error):
        self._metadata_failed(error)

    def _metadata_failed(self, error):
        if self._metadata_prepare_error is None:
            self._metadata_prepare_error = error
        self._owner_failed = True
        self._compute_failed(error)

    def _build_cpu_ticket(self, step, sparse_features, sparse_batch):
        if step >= self._total_steps:
            raise RuntimeError("Owner PS prepared beyond the declared step count")
        start = time.perf_counter()
        raw = self._raw_sparse_ids(sparse_features, sparse_batch)
        from .canonical_ps import make_ticket_with_inverse
        local_ticket, inverse = make_ticket_with_inverse(
            raw.numpy(), self._offsets.numpy(), self._owner_table_types)
        ticket = (torch.from_numpy(local_ticket.keys), torch.from_numpy(inverse), raw.numel())
        return ticket, local_ticket, (time.perf_counter() - start) * 1e3

    def _adopt_cpu_ticket(self, step, ticket, future, row, build_ms):
        self._check_prepared_step(step)
        if future is None:
            raise RuntimeError("Owner ticket lacks its metadata Future")
        self._owner_metadata[step] = future
        if step + 1 < self._total_steps:
            self._metadata[step] = future
        self._queue_prepared_read(step, ticket[0])
        self._prepared += 1
        row["lookup_ids_build_ms"] = build_ms
        return ticket

    def _exchange_key_counts(self, step, ticket):
        start = time.perf_counter()
        header = torch.tensor([step, len(ticket.keys), self._total_steps, self._layout_digest,
                               ticket.batch_size, ticket.features], dtype=torch.int64)
        headers = [torch.empty_like(header) for _ in range(self._world)]
        dist.all_gather(headers, header, group=self._metadata_group)
        for peer in headers:
            if (int(peer[0]) != step or int(peer[2]) != self._total_steps
                    or int(peer[3]) != self._layout_digest or int(peer[5]) != ticket.features
                    or not 0 <= int(peer[4]) <= 1024
                    or not 0 <= int(peer[1]) <= int(peer[4]) * ticket.features):
                raise RuntimeError("Owner metadata layout/step/count mismatch")
        sizes = [int(peer[1]) for peer in headers]
        width = max(sizes)
        payload = torch.zeros((width, 2), dtype=torch.int64)
        payload[:len(ticket.keys), 0] = torch.from_numpy(ticket.keys)
        payload[:len(ticket.keys), 1] = torch.from_numpy(ticket.counts)
        peers = [torch.empty_like(payload) for _ in range(self._world)]
        if width:
            dist.all_gather(peers, payload, group=self._metadata_group)
        for rank, peer in enumerate(peers):
            counts = peer[:sizes[rank], 1]
            if (bool((counts < 1).any()) or bool((counts > int(headers[rank][4])).any())
                    or int(counts.sum()) != int(headers[rank][4]) * ticket.features):
                raise RuntimeError("Owner metadata occurrence counts disagree with batch shape")
        exchanged = time.perf_counter()
        from .canonical_ps import OwnerExchangePlan
        plan = OwnerExchangePlan(len(ticket.keys))
        self._build_owner_plan(plan, peers, sizes)
        dirty = torch.from_numpy(np.unique(np.concatenate(
            [peer[:count, 0].numpy() for peer, count in zip(peers, sizes)])))
        peer_bytes = (header.numel() + 2 * width) * 8
        return dirty, {
            "ps_pipeline_metadata_step": step,
            "ps_pipeline_metadata_ms": (exchanged - start) * 1e3,
            "ps_pipeline_metadata_sent_bytes": peer_bytes,
            "ps_pipeline_metadata_received_bytes": peer_bytes,
            "ps_pipeline_local_dirty_rows": len(ticket.keys),
            "ps_pipeline_global_dirty_rows": dirty.numel(),
        }, plan, (time.perf_counter() - exchanged) * 1e3

    def _build_cpu_metadata(self, step, source):
        if self._metadata_stopping or self._metadata_prepare_error is not None:
            raise self._metadata_prepare_error or RuntimeError("Owner metadata producer stopped")
        try:
            ticket, local, build_ms = self._build_cpu_ticket(step, None, source)
            # Same single Gloo worker: call directly, never submit and self-wait.
            metadata = self._exchange_key_counts(step, local)
            plan = metadata[2]
            tensors = (*ticket[:2], metadata[0], *plan.owner)
            if any(x.device.type != "cpu" for x in tensors):
                raise RuntimeError("Metadata lookahead produced a non-CPU tensor")
            # Logical payload bytes; aliases can be counted twice, not allocator RSS.
            payload_bytes = sum(x.numel() * x.element_size() for x in tensors)
            return (self._session, self._layout_digest, step, source,
                    ticket, metadata, build_ms, payload_bytes)
        except BaseException as error:
            self._metadata_failed(error)
            raise

    def queue_cpu_metadata(self, step, source):
        try:
            if (self._closed or self._owner_failed
                    or self._metadata_stopping or self._metadata_prepare_error is not None):
                raise self._metadata_prepare_error or RuntimeError("Owner metadata queue is unavailable")
            if (step != self._metadata_enqueued or step >= self._total_steps
                    or step > self._current_step + self.metadata_lookahead + 1):
                raise RuntimeError("Owner metadata queue exceeded its ordered bounded window")
            if (source.device.type != "cpu" or source.dtype != torch.int64
                    or source.ndim != 2 or source.shape[1] != len(self._keys)):
                raise ValueError("Owner metadata lookahead requires immutable CPU int64 IDs[B,F]")
            future = self._metadata_worker.submit(self._build_cpu_metadata, step, source)
            self._metadata_enqueued += 1
            self._metadata_queue_hwm = max(self._metadata_queue_hwm, self._metadata_enqueued-self._prepared)
            return _CPUPreparation(step, source, future)
        except BaseException as error:
            self._metadata_failed(error)
            raise

    def adopt_cpu_metadata(self, step, handle, row):
        try:
            if self._owner_failed or self._metadata_prepare_error is not None:
                raise self._metadata_prepare_error or RuntimeError("GPU owner session has failed")
            self._check_prepared_step(step)  # The original one-next value-read guard.
            if not isinstance(handle, _CPUPreparation) or handle.step != step:
                raise RuntimeError("Owner metadata handle has the wrong step")
            start = time.perf_counter()
            session, layout, actual_step, source, ticket, metadata, build_ms, size = handle.future.result()
            if ((session, layout, actual_step) != (self._session, self._layout_digest, step)
                    or source is not handle.source):
                raise RuntimeError("Owner metadata identity differs from its CPU batch")
            ready = Future()
            ready.set_result(metadata)  # Completed real calculation, never a fabricated commit proof.
            row.update(ps_metadata_adopt_wait_ms=(time.perf_counter()-start)*1e3,
                       ps_metadata_ticket_build_ms=build_ms,
                       ps_metadata_plan_payload_bytes=size)
            return self._adopt_cpu_ticket(step, ticket, ready, row, build_ms)
        except BaseException as error:
            self._metadata_failed(error)
            raise

    def _compute_failed(self, error):
        self._owner_failed = True
        if self._compute_error is None:
            self._compute_error = error

    def _record_compute_state(self, row):
        native_depth = self._accepted_step - self._retired_step
        successor_depth = self._candidate_step - self._accepted_step
        if not (0 <= native_depth <= 1 and 0 <= successor_depth <= 1):
            raise RuntimeError("Owner compute-ahead depth exceeded")
        depth = self._candidate_step - self._retired_step
        row.update(ps_owner_candidate_step=self._candidate_step,
                   ps_owner_native_submitted_step=self._accepted_step,
                   ps_owner_retired_step=self._retired_step,
                   ps_owner_unretired_depth=depth,
                   ps_owner_unretired_native_depth=native_depth,
                   ps_owner_retiring_holders=int(self._retiring_owner is not None),
                   ps_owner_max_retiring_holders=max(row.get("ps_owner_max_retiring_holders", 0),
                                                    int(self._retiring_owner is not None)),
                   ps_owner_max_unretired_depth=max(row.get("ps_owner_max_unretired_depth", 0), depth))

    def _retire_previous(self, step):
        if step == 0:
            if self._commit is not None or self._retired_step != -1:
                raise RuntimeError("Owner bootstrap has an unexpected commit")
            return
        if self._commit is None or self._commit_step != step - 1:
            raise RuntimeError("Owner retirement lacks the exact previous global commit")
        self._commit.result()
        self._retired_step = step - 1
        if self._retiring_owner is not None:
            if self._retiring_owner[0] != step - 1:
                raise RuntimeError("Owner retirement holder has the wrong step")
            self._retiring_owner = None  # ACK covers its candidate/status D2H and producers.

    def submit_sparse_grad(self, ids, grads):
        if self._owner_failed:
            raise RuntimeError("GPU owner session has failed")
        try:
            return super().submit_sparse_grad(ids, grads)
        except BaseException as error:
            # Includes executor failure after native accepted the handle.
            self._compute_failed(error)
            raise

    def _wait_pending(self):
        if self._owner_failed:
            raise self._compute_error or RuntimeError("GPU owner session has failed")
        try:
            if self._accepted_step >= 0:
                if (self._commit is None
                        or self._commit_step != self._accepted_step):
                    raise RuntimeError("Owner final retirement lacks a tracked global commit")
                self._commit.result()
                self._retired_step = self._commit_step
                self._retiring_owner = None
            if self._current_metrics is not None:
                self._record_compute_state(self._current_metrics)
        except BaseException as error:
            self._compute_failed(error)
            raise

    def after_step_sync(self, row):
        super().after_step_sync(row)
        self._record_compute_state(row)

    def _build_owner_plan(self, plan, peers, sizes):
        columns = [peer[:size, column].contiguous()
                   for peer, size in zip(peers, sizes) for column in (0, 1)]
        operation = self.kv_client.ops.owner_merged_forward_plan
        owner = operation(
            *columns, self._owner_boundaries, self._owner_types, self._owner_tag, self._rank)
        if len(owner) != 9:
            raise RuntimeError("Unexpected owner plan schema")
        if owner[0].numel() != plan.local_unique_count:
            raise RuntimeError("Owner routing disagrees with the local unique key count")
        plan.owner = owner

    def on_batch_prepared(self, step, sparse_features, sparse_batch, row):
        if self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        try:
            self._check_prepared_step(step)
            start = time.perf_counter()
            ticket, local, build_ms = self._build_cpu_ticket(step, sparse_features, sparse_batch)
            future = self._metadata_worker.submit(self._exchange_key_counts, step, local)
            result = self._adopt_cpu_ticket(step, ticket, future, row, build_ms)
            row["lookup_ids_build_ms"] = (time.perf_counter() - start) * 1e3
            return result
        except BaseException as error:
            self._compute_failed(error)
            raise

    def _queue_prepared_read(self, step, ids):
        if self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        if step == 0:
            self._read = self._worker.submit(
                self._pull_owned_step, step, self._owner_metadata[step], self._commit)
        else:
            if self._pending_owner_step is not None:
                raise RuntimeError("Unified owner prepares only one next step")
            # A speculative input Future is not a committed-version proof.
            if step > 1:
                if self._commit_step != step - 2 or self._commit is None:
                    raise RuntimeError("Owner clean read lacks its exact prior global commit")
                previous_read = self._commit
            else:
                previous_read = self._read or self._consumed_read
            if previous_read is None:
                raise RuntimeError("Current complete owner read was not published")
            self._split_clean = self._worker.submit(
                self._read_owner_clean, step, previous_read,
                self._owner_metadata[step - 1], self._owner_metadata[step])
            self._pending_owner_step = step
            # Only owner_metadata is consumed by backward in this mode.
            self._metadata.pop(step - 1, None)

    def _require_next_read(self):
        if self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        if (self._submitted + 1 < self._total_steps
                and self._pending_owner_step != self._submitted + 1):
            raise RuntimeError("Prepare next owner metadata before current submission")
        if self._submitted + 1 < self._total_steps and self._split_clean is None:
            raise RuntimeError("Prepare next owner clean read before current submission")

    def _queue_next_read(self, commit):
        step = self._submitted - 1
        if self._accepted_step != step or self._commit is not commit:
            raise RuntimeError("Owner commit tracking differs from native acceptance")
        self._commit_step = step  # Base has already enqueued W(step).
        if self._pending_owner_step is not None:
            if (self._pending_owner_step != self._submitted
                    or self._compute_read_step != self._submitted or self._read is None):
                raise RuntimeError("Owner speculative read was not queued before commit wait")
            self._pending_owner_step = self._compute_read_step = None

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
        if commit is not None:
            raise RuntimeError("Speculative input must not wait on current publication")
        waited = 0.0
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
                if hasattr(error, "add_note"):
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
            "ps_owner_bootstrap_rows": values.shape[0],
        }

    def _stage_rows(self, step, ticket, values, row_count, row):
        from .owner_forward_gpu import ForwardGPUOwnerExchange as ForwardExchange
        plan = self._owner_metadata[step].result()[2].owner
        permutation, keys, metadata, send, recv, index, fsend, frecv, reorder = plan
        if reorder.numel() != ticket[0].numel():
            raise RuntimeError("Owner forward reorder differs from local ticket")
        # Compute-ahead moved the previous helper/plan into a bounded retiring slot.
        # The acknowledged path still proves prior GLOBAL ACK at this point.
        self._owner_exchange = None
        self._owner_plan = plan
        start = time.perf_counter()
        source_refs = ()
        try:
            if isinstance(values, _OwnerHandoffRows):
                merge_start = time.perf_counter()
                values, source_refs = self._merge_owner_handoff(step, values, keys)
                row["ps_owner_handoff_merge_enqueue_ms"] = (time.perf_counter() - merge_start) * 1e3
                row["ps_owner_handoff_source_refs"] = len(source_refs)
                row["ps_owner_handoff_source_bytes"] = sum(x.numel() * x.element_size() for x in source_refs)
                row["ps_owner_handoff_source_cpu_bytes"] = sum(
                    x.numel() * x.element_size() for x in source_refs if x.device.type == "cpu")
                row["ps_owner_handoff_source_cuda_bytes"] = sum(
                    x.numel() * x.element_size() for x in source_refs if x.device.type == "cuda")
            self._owner_exchange = ForwardExchange(
                permutation, keys, metadata, send, recv, values,
                forward_send_index=index, forward_reorder=reorder,
                forward_send_splits=fsend, forward_recv_splits=frecv,
                process_group=dist.group.WORLD, device=self.device, source_refs=source_refs)
            self._current_rows = self._owner_exchange.forward_rows()
        except BaseException as error:
            try:
                torch.cuda.synchronize(self.device)
            except BaseException as cleanup:
                if hasattr(error, "add_note"):
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
        if self._owner_failed:
            raise RuntimeError("GPU owner session has failed")
        try:
            if self._retiring_owner is not None:
                raise RuntimeError("Owner retirement holder is still occupied")
            if step:
                if self._owner_exchange is None or self._current_step != step - 1:
                    raise RuntimeError("Owner compute-ahead lacks preceding GPU staging")
                self._retiring_owner = (step - 1, self._owner_exchange)
                self._owner_exchange = None
            row.update(ps_owner_retire_wait_ms=0.0, ps_owner_max_unretired_depth=0)
            self._record_compute_state(row)
            super().before_lookup(step, sparse_features, ticket, row)
            row["ps_pipeline_read_bytes"] = row["ps_owner_old_read_bytes"]
        except BaseException as error:
            self._compute_failed(error)
            raise

    def _submit_gradients(self, ids, grads):
        if self._owner_failed:
            raise RuntimeError("Unified owner session has failed")
        try:
            return self._submit_owner_merged(grads)
        except BaseException as error:
            self._compute_failed(error)
            raise

    def _submit_owner_merged(self, grads):
        start = time.perf_counter()
        if self._owner_exchange is None or self._owner_plan is None:
            raise RuntimeError("Owner submission lacks completed forward staging")
        permutation, keys, metadata, send, recv = self._owner_plan[:5]
        pull_ms = self._current_metrics["ps_owner_pull_ms"]
        read_done = start
        candidates, status = self._owner_exchange.run(grads, self.lr)
        if self._submitted + 1 < self._total_steps:
            if self._owner_candidate is not None:
                raise RuntimeError("Owner candidate already awaits handoff")
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            self._owner_candidate = (self._session, self._submitted, self._rank,
                                     keys, candidates, ready)
        enqueued = time.perf_counter()
        self._candidate_step = self._submitted  # Kernel enqueued, not yet a retired result.
        self._record_compute_state(self._current_metrics)
        retire_start = time.perf_counter()
        self._retire_previous(self._submitted)
        self._current_metrics["ps_owner_retire_wait_ms"] = (time.perf_counter() - retire_start) * 1e3
        publish_start = time.perf_counter()
        handle = self.kv_client.sparse_owner_replace_async(
            self.master_table_name, keys, candidates, status, session=self._session,
            step=self._submitted, rank=self._rank, world=self._world, lr=self.lr)
        self._track_commit(handle)
        if self._pending_owner_step is not None:
            if (self._pending_owner_step != self._submitted + 1
                    or self._split_clean is None or self._owner_candidate is None):
                raise RuntimeError("Owner compute-ahead lacks a prepared successor")
            self._read = self._worker.submit(self._finish_owner_handoff,
                self._pending_owner_step, self._split_clean, None, self._owner_candidate)
            self._compute_read_step = self._pending_owner_step
            self._split_clean = self._owner_candidate = None
        self._record_compute_state(self._current_metrics)
        self._current_metrics.update(
            ps_owner_read_wait_ms=(read_done - start) * 1e3,
            ps_owner_pull_ms=pull_ms,
            ps_owner_exchange_enqueue_ms=(enqueued - read_done) * 1e3,
            ps_owner_commit_submit_ms=(time.perf_counter() - publish_start) * 1e3,
            ps_owner_peer_send_bytes=int(send[1 - self._rank]) * 128 * 4,
            ps_owner_peer_recv_bytes=int(recv[1 - self._rank]) * 128 * 4,
            ps_owner_candidate_bytes=candidates.numel() * 4)
        return handle

    def _valid_gradient_shape(self, ids, grads):
        del ids
        return (grads.ndim == 2 and grads.shape[1] == self.embedding_dim
                and grads.dtype == torch.float32)

    def _release_step(self, step):
        self._owner_metadata.pop(step, None)

    @property
    def table_types(self):
        return self._owner_table_types

    def _close_commit_session(self, abort):
        self.kv_client.sparse_owner_replace_close(
            self.master_table_name, session=self._session,
            step=self._accepted_step,
            rank=self._rank, world=self._world, lr=self.lr, abort=abort)

    def shutdown(self, *, flush=True):
        if self._closed:
            return
        self._metadata_stopping = True
        error = self._compute_error or self._metadata_prepare_error
        if not flush or error is not None or self._submission_error is not None:
            # Abort and drain before synchronizing GPU producers or releasing rows.
            try:
                self._finish_commits(abort=True)
            except BaseException as cleanup:
                if error is None:
                    error = cleanup
                else:
                    if hasattr(error, "add_note"):
                        error.add_note("Owner commit cleanup also failed: " + repr(cleanup))
        try:
            if flush and error is None:
                # Observe a failed commit/read before secondary cleanup errors.
                self._wait_pending()
                if self._read is not None:
                    self._read.result()
        except BaseException as exc:
            error = exc
        try:
            # Read futures may own merge sources: drain before base clears them.
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                if hasattr(error, "add_note"):
                    error.add_note("Owner handoff cleanup also failed: " + repr(exc))
        if self._retiring_owner is not None:
            for owner in self._retiring_owner[1:]:
                try:
                    if owner is not None:
                        owner.drain()
                except BaseException as cleanup:
                    if error is None:
                        error = cleanup
                    else:
                        if hasattr(error, "add_note"):
                            error.add_note("Retiring owner cleanup also failed: " + repr(cleanup))
        try:
            super().shutdown(flush=flush and error is None)
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                if hasattr(error, "add_note"):
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
                if hasattr(error, "add_note"):
                    error.add_note("GPU owner cleanup also failed: " + repr(exc))
        finally:
            self._owner_exchange = self._owner_plan = None
            self._pending_owner_step = None
            self._owner_candidate = None
            self._retiring_owner = None
            self._owner_metadata.clear()
        if error is not None:
            raise error
