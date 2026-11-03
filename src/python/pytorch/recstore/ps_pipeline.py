"""One-step PS pipeline with committed reads and no parameter cache."""
from __future__ import annotations

import hashlib
import os
import secrets
from typing import NamedTuple
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


class _DeviceSplitRows(NamedTuple):
    clean_pos: torch.Tensor
    dirty_pos: torch.Tensor
    clean_values: torch.Tensor
    dirty_values: torch.Tensor
    row_count: int


class PSPipeline:
    """Prepare local IDs, then read the next batch after global SGD commits.

    One CPU worker owns commit waits and reads. Dependency splitting uses a
    separate Gloo worker to identify rows safe to read before the current
    commit. GPU embedding values belong only to the consumed batch.
    """

    owner_merged_grad = False
    depth = 1
    desired_buffer_size = 0

    def __init__(self, embedding_module, kv_client, *, table_offsets,
                 master_table_name, embedding_dim, device, lr=0.01,
                 validation_state=False, dependency_split=False, total_steps=None,
                 owner_table_types=None):
        self.embedding_module = embedding_module
        self.kv_client = kv_client
        self.master_table_name = master_table_name
        self.embedding_dim = int(embedding_dim)
        self.device = torch.device(device)
        self.lr = float(lr)
        self._keys = list(table_offsets)
        self._offsets = torch.tensor(list(table_offsets.values()), dtype=torch.int64)
        self._world = dist.get_world_size() if dist.is_initialized() else 1
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self.dependency_split = bool(dependency_split)
        self._device_merge = os.environ.get("RECSTORE_PS_DEVICE_MERGE") == "1"
        if self._device_merge and not (self.dependency_split and self.device.type == "cuda"):
            raise ValueError("Device row merge requires dependency-split CUDA PS")
        self._total_steps = total_steps
        self.owner_updates = owner_table_types is not None
        self._owner_table_types = tuple(owner_table_types or ())
        self._owner_metadata = {}
        self._current_metrics = None
        self._gradient_bytes = 0
        if self.owner_updates:
            if not self.owner_merged_grad:
                raise ValueError("The count-exchange ticket mode requires the merged GPU owner pipeline")
            if (self._world != 2 or self.embedding_dim != 128 or self.device.type != "cuda"
                    or len(self._owner_table_types) != len(self._keys)
                    or any(kind not in (0, 1) for kind in self._owner_table_types)
                    or not isinstance(total_steps, int) or isinstance(total_steps, bool)
                    or total_steps < 1):
                raise ValueError("Owner PS requires world2, CUDA D128 and an explicit table layout")
            layout = (self._offsets.numpy().tobytes() + bytes(self._owner_table_types)
                      + "\0".join(self._keys).encode("utf-8"))
            self._layout_digest = int.from_bytes(hashlib.sha256(layout).digest()[:8], "little") & (2**63 - 1)
        if self.dependency_split and (self._world != 2 or not isinstance(total_steps, int)
                                      or isinstance(total_steps, bool) or total_steps < 1):
            raise ValueError("Dependency-split pull requires world=2 and positive total_steps")
        session = torch.tensor([secrets.randbelow(2**63 - 1) + 1],
                               dtype=torch.int64, device=self.device)
        if self._world > 1:
            dist.broadcast(session, src=0)
        self._session = int(session.item())
        self.metadata_group_init_ms = 0.0
        self._metadata_group = self._metadata_worker = None
        self._metadata = {}
        self._split_clean = None
        self._consumed_read = None
        self._read = None
        self._pending_read_ids = None
        self._commit = None
        self._pending_commits = {}
        self._accepted_step = -1
        self._commit_session_closed = False
        self._submission_error = None
        self._prepared = 0
        self._submitted = 0
        self._current_step = -1
        self._current_ticket = None
        self._current_rows = None
        self._current_inverse_cpu = None
        self._current_inverse = None
        self._closed = False
        self._updated_ids = set() if validation_state else None
        try:
            if self.dependency_split or self.owner_updates:
                start = time.perf_counter()
                self._metadata_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
                self.metadata_group_init_ms = (time.perf_counter() - start) * 1e3
                self._metadata_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ps-ids")
            self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ps-pipeline")
        except BaseException:
            # A failed constructor is never returned to the runner for shutdown.
            # Preserve the construction error even if releasing resources fails.
            if self._metadata_worker is not None:
                try:
                    self._metadata_worker.shutdown(wait=True, cancel_futures=True)
                except BaseException:
                    pass
            if self._metadata_group is not None:
                try:
                    dist.destroy_process_group(self._metadata_group)
                except BaseException:
                    pass
            raise

    def _pull(self, ids, previous_commit):
        waited = previous_commit.result() if previous_commit is not None else 0.0
        start = time.perf_counter()
        pull_into = getattr(self.kv_client, "pull_into", None)
        if ids.numel() and callable(pull_into):
            values = torch.empty((ids.numel(), self.embedding_dim),
                                 dtype=torch.float32,
                                 pin_memory=self.device.type == "cuda")
            pull_into(self.master_table_name, ids, values)
        else:
            values = (self.kv_client.pull(self.master_table_name, ids)
                      if ids.numel() else torch.empty((0, self.embedding_dim)))
        if values.shape != (ids.numel(), self.embedding_dim) or values.dtype != torch.float32:
            raise RuntimeError("PS pipeline received invalid embedding shape or dtype")
        if values.device.type != "cpu":
            raise RuntimeError("PS pipeline reads must return CPU staging tensors")
        if self.device.type == "cuda" and not values.is_pinned():
            values = values.pin_memory()
        return values, waited, (time.perf_counter() - start) * 1e3

    def _exchange_dirty_ids(self, step, ids):
        start = time.perf_counter()
        header = torch.tensor([step, ids.numel(), self._total_steps], dtype=torch.int64)
        headers = [torch.empty_like(header) for _ in range(self._world)]
        dist.all_gather(headers, header, group=self._metadata_group)
        if any(int(h[0]) != step or int(h[2]) != self._total_steps for h in headers):
            raise RuntimeError("Dependency metadata step/count mismatch")
        counts = [int(h[1]) for h in headers]
        if any(count < 0 for count in counts):
            raise RuntimeError("Invalid dependency metadata length")
        width = max(counts)
        payload = torch.zeros(width, dtype=torch.int64)
        payload[:ids.numel()].copy_(ids)
        peers = [torch.empty_like(payload) for _ in range(self._world)]
        # All ranks agree on width, including an all-empty step.
        if width:
            dist.all_gather(peers, payload, group=self._metadata_group)
        dirty = torch.from_numpy(np.unique(np.concatenate(
            [peer[:count].numpy() for peer, count in zip(peers, counts)])))
        peer_bytes = (header.numel() + width) * 8 * (self._world - 1)
        return dirty, {
            "ps_pipeline_metadata_step": step,
            "ps_pipeline_metadata_ms": (time.perf_counter() - start) * 1e3,
            "ps_pipeline_metadata_sent_bytes": peer_bytes,
            "ps_pipeline_metadata_received_bytes": peer_bytes,
            "ps_pipeline_local_dirty_rows": ids.numel(),
            "ps_pipeline_global_dirty_rows": dirty.numel(),
        }

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

    def _build_owner_plan(self, plan, peers, sizes):
        raise NotImplementedError("Owner routing requires the GPU owner pipeline")

    @staticmethod
    def _partition_clean_rows(ids, dirty_ids):
        query, dirty = ids.numpy(), dirty_ids.numpy()
        positions = np.searchsorted(dirty, query)
        matches = np.zeros(query.shape, dtype=np.bool_)
        valid = positions < dirty.size
        matches[valid] = dirty[positions[valid]] == query[valid]
        dirty_mask = torch.from_numpy(matches)
        clean_pos = torch.nonzero(~dirty_mask, as_tuple=True)[0]
        dirty_pos = torch.nonzero(dirty_mask, as_tuple=True)[0]
        return clean_pos, dirty_pos, torch.from_numpy(positions)

    def _read_clean(self, ids, previous_read, metadata):
        # The previous read proves commit(k-1) succeeded; metadata is independent
        # of native commit(k), so neither wait depends on a later I/O task.
        previous_read.result()
        start = time.perf_counter()
        dirty_ids, stats = metadata.result()[:2]
        stats = dict(stats)
        stats["ps_pipeline_metadata_wait_ms"] = (time.perf_counter() - start) * 1e3
        start = time.perf_counter()
        # Metadata is already sorted. Avoid sorting both sets again in isin.
        clean_pos, dirty_pos, _ = self._partition_clean_rows(ids, dirty_ids)
        stats["ps_pipeline_split_plan_ms"] = (time.perf_counter() - start) * 1e3
        values, _, elapsed = self._pull(ids[clean_pos], None)
        stats.update(ps_pipeline_clean_pull_ms=elapsed,
                     ps_pipeline_clean_rows=clean_pos.numel(),
                     ps_pipeline_dirty_rows=dirty_pos.numel(),
                     ps_pipeline_clean_read_bytes=values.numel() * values.element_size())
        return ids, clean_pos, dirty_pos, values, stats

    def _finish_split_read(self, clean, commit):
        ids, clean_pos, dirty_pos, clean_values, stats = clean.result()
        waited = commit.result()  # Global ACK over every PS shard.
        dirty_values, _, elapsed = self._pull(ids[dirty_pos], None)
        stats["ps_pipeline_dirty_pull_ms"] = elapsed
        stats["ps_pipeline_dirty_read_bytes"] = dirty_values.numel() * dirty_values.element_size()
        start = time.perf_counter()
        if (self._device_merge and ids.numel() >= 4096
                and clean_pos.numel() and dirty_pos.numel()):
            # Preserve owned pinned parts until their current-stream copies finish.
            # Small/tail batches keep the lower-launch-overhead CPU path.
            values = _DeviceSplitRows(clean_pos, dirty_pos, clean_values, dirty_values,
                                      ids.numel())
        elif not dirty_pos.numel():
            values = clean_values
        elif not clean_pos.numel():
            values = dirty_values
        else:
            values = torch.empty((ids.numel(), self.embedding_dim), dtype=torch.float32,
                                 pin_memory=self.device.type == "cuda")
            output = values.numpy()
            output[clean_pos.numpy()] = clean_values.numpy()
            output[dirty_pos.numpy()] = dirty_values.numpy()
        stats["ps_pipeline_split_merge_ms"] = (time.perf_counter() - start) * 1e3
        return values, waited, stats["ps_pipeline_clean_pull_ms"] + elapsed, stats

    def _check_prepared_step(self, step):
        if (self._closed or step != self._prepared or
                (not self.dependency_split and (self._read is not None
                 or self._pending_read_ids is not None))):
            raise RuntimeError("PS pipeline requires one ordered prepared batch")
        if self.dependency_split:
            if (step >= self._total_steps or self._split_clean is not None or
                    (step != 0 and step != self._submitted + 1)):
                raise RuntimeError("Dependency split prepares one next batch before current submission")
        elif step != self._submitted and not (
                step == self._submitted + 1 and self._current_step == self._submitted):
            raise RuntimeError("PS pipeline prepares at most the next batch")

    def _build_cpu_ticket(self, step, sparse_features, sparse_batch):
        """Pure ticket construction; no read, collective, or preparation counters."""
        if sparse_features is not None:
            if list(sparse_features.keys()) != self._keys:
                raise ValueError("PS pipeline feature order differs from the table layout")
        elif sparse_batch is None:
            raise ValueError("PS pipeline needs a sparse tensor or keyed features")
        start = time.perf_counter()
        if sparse_batch is not None:
            if sparse_batch.ndim != 2 or sparse_batch.shape[1] != len(self._keys):
                raise ValueError("PS pipeline requires one ID per feature per sample")
            raw = sparse_batch.detach().to(device="cpu", dtype=torch.int64).t()
        else:
            lengths = sparse_features.lengths().cpu()
            raw = sparse_features.values().detach().to(device="cpu", dtype=torch.int64)
            if raw.numel() % len(self._keys) or lengths.numel() != raw.numel() or not bool((lengths == 1).all()):
                raise ValueError("PS pipeline requires one ID per feature per sample")
            raw = raw.reshape(len(self._keys), -1)
        local_ticket = None
        if self.owner_updates:
            if step >= self._total_steps:
                raise RuntimeError("Owner PS prepared beyond the declared step count")
            from .canonical_ps import make_ticket_with_inverse
            local_ticket, inverse_numpy = make_ticket_with_inverse(
                raw.numpy(), self._offsets.numpy(), self._owner_table_types)
            ids = torch.from_numpy(local_ticket.keys)
            inverse = torch.from_numpy(inverse_numpy)
        else:
            raw = (raw + self._offsets[:, None]).reshape(-1)
            ids, inverse = torch.unique(raw, return_inverse=True)
        ticket = ids, inverse, raw.numel()
        return ticket, local_ticket, (time.perf_counter() - start) * 1e3

    def _adopt_cpu_ticket(self, step, ticket, future, row, build_ms):
        self._check_prepared_step(step)
        if self.owner_updates:
            if future is None:
                raise RuntimeError("Owner ticket lacks its metadata Future")
            self._owner_metadata[step] = future
            if self.dependency_split and step + 1 < self._total_steps:
                self._metadata[step] = future
        self._queue_prepared_read(step, ticket[0])
        self._prepared += 1
        row["lookup_ids_build_ms"] = build_ms
        return ticket

    def on_batch_prepared(self, step, sparse_features, sparse_batch, row):
        self._check_prepared_step(step)
        start = time.perf_counter()
        ticket, local_ticket, build_ms = self._build_cpu_ticket(step, sparse_features, sparse_batch)
        future = None
        if self.owner_updates:
            # Counts remain required for the final step, without another value read.
            future = self._metadata_worker.submit(self._exchange_key_counts, step, local_ticket)
        result = self._adopt_cpu_ticket(step, ticket, future, row, build_ms)
        row["lookup_ids_build_ms"] = (time.perf_counter() - start) * 1e3
        return result

    def _queue_prepared_read(self, step, ids):
        if self.dependency_split:
            if not self.owner_updates and step + 1 < self._total_steps:
                self._metadata[step] = self._metadata_worker.submit(self._exchange_dirty_ids, step, ids)
            if step == 0:
                self._read = self._worker.submit(self._pull, ids, None)
            else:
                # Consume publishes _consumed_read before clearing _read.
                # Read _read once so concurrent consumption cannot replace a second lookup.
                previous_read = self._read
                if previous_read is None:
                    previous_read = self._consumed_read
                if previous_read is None:
                    raise RuntimeError("Current batch read was not published")
                self._split_clean = self._worker.submit(
                    self._read_clean, ids, previous_read, self._metadata.pop(step - 1))
        elif step == self._submitted:
            self._read = self._worker.submit(self._pull, ids, self._commit)
        else:
            # IDs can be planned early, but values must wait for global SGD.
            self._pending_read_ids = ids

    def _require_next_read(self):
        if (self.dependency_split and self._submitted + 1 < self._total_steps
                and self._split_clean is None):
            raise RuntimeError("Prepare next dependency-split batch before submitting current SGD")

    def _queue_next_read(self, commit):
        if self.dependency_split and self._split_clean is not None:
            self._read = self._worker.submit(self._finish_split_read, self._split_clean, commit)
            self._split_clean = None
        if self._pending_read_ids is not None:
            self._read = self._worker.submit(self._pull, self._pending_read_ids, commit)
            self._pending_read_ids = None

    def _stage_rows(self, step, ticket, values, row_count, row):
        if isinstance(values, _DeviceSplitRows):
            self._current_rows = torch.empty((row_count, self.embedding_dim),
                                             dtype=torch.float32, device=self.device)
            for positions, part in ((values.clean_pos, values.clean_values),
                                    (values.dirty_pos, values.dirty_values)):
                index = positions.pin_memory().to(self.device, non_blocking=True)
                transferred = part.to(self.device, non_blocking=True)
                # Positions are unique and disjoint, so this only copies FP32 bits.
                self._current_rows.index_copy_(0, index, transferred)
        else:
            self._current_rows = values.to(self.device, non_blocking=True)

    def before_lookup(self, step, sparse_features, ticket, row):
        del sparse_features
        if self._read is None or step != self._current_step + 1:
            raise RuntimeError("PS pipeline consumed an unprepared or unordered batch")
        start = time.perf_counter()
        result = self._read.result()
        if self.dependency_split:
            self._consumed_read = self._read
        values, commit_wait, pull_ms = result[:3]
        row_count = values.row_count if isinstance(values, _DeviceSplitRows) else values.shape[0]
        read_bytes = row_count * self.embedding_dim * 4
        row["ps_pipeline_device_merge"] = int(isinstance(values, _DeviceSplitRows))
        if self.dependency_split:
            for name in ("metadata_ms", "metadata_sent_bytes", "metadata_received_bytes",
                         "local_dirty_rows", "global_dirty_rows", "metadata_wait_ms",
                         "split_plan_ms", "clean_pull_ms", "dirty_pull_ms", "clean_rows",
                         "dirty_rows", "clean_read_bytes", "dirty_read_bytes", "split_merge_ms"):
                row["ps_pipeline_" + name] = 0
            row["ps_pipeline_metadata_step"] = -1
            row["ps_pipeline_dirty_pull_ms"] = pull_ms
            row["ps_pipeline_dirty_rows"] = row_count
            row["ps_pipeline_dirty_read_bytes"] = read_bytes
        if len(result) == 4:
            row.update(result[3])
        self._read = None
        row["ps_pipeline_read_wait_ms"] = (time.perf_counter() - start) * 1e3
        row["ps_pipeline_commit_wait_ms"] = commit_wait
        row["ps_pipeline_pull_ms"] = pull_ms
        row["ps_pipeline_read_bytes"] = read_bytes
        row["ps_pipeline_cache_bytes"] = 0
        # Use sample-major indices for both lookup output and pooled gradients.
        # Retain pinned source storage until the next committed batch is read.
        batch = ticket[2] // len(self._keys)
        inverse = ticket[1].reshape(len(self._keys), batch).t().reshape(-1)
        if self.device.type == "cuda":
            inverse = inverse.pin_memory()
        self._current_inverse_cpu = inverse
        self._current_inverse = inverse.to(self.device, non_blocking=True)
        self._stage_rows(step, ticket, values, row_count, row)
        self._current_step = step
        self._current_ticket = ticket
        self._current_metrics = row

    def lookup(self, sparse_features):
        del sparse_features
        from torchrec_kv.EmbeddingBag import KeyedTensor
        if self._current_rows is None:
            raise RuntimeError("PS pipeline lookup requires before_lookup")
        values = self._current_rows.index_select(0, self._current_inverse)
        batch = values.size(0) // len(self._keys)
        return KeyedTensor(keys=self._keys, values=values.reshape(batch, len(self._keys) * self.embedding_dim),
                           length_per_key=[self.embedding_dim] * len(self._keys))

    def submit_sparse_grad(self, ids, grads):
        """Submit exactly once after the current batch's row gradients are ready."""
        if self._closed or self._submission_error is not None:
            raise self._submission_error or RuntimeError("PS pipeline is closed")
        if self._current_step != self._submitted or self._current_ticket is None:
            raise RuntimeError("PS pipeline requires exactly one SGD submission per batch")
        expected = self._current_ticket[0]
        if ids is not expected and (ids.device.type != "cpu" or not torch.equal(ids, expected)):
            raise ValueError("PS pipeline gradients do not match the prepared IDs")
        if (grads.ndim != 2 or grads.shape[1] != self.embedding_dim
                or grads.dtype != torch.float32
                or (not self.owner_updates and grads.shape[0] != ids.numel())):
            raise ValueError("PS pipeline gradients must be FP32 with the prepared shape")
        self._require_next_read()
        # Reap completed waits without blocking compute-ahead on its predecessor.
        for handle, future in list(self._pending_commits.items()):
            if future is not None and future.done() and not future.cancelled():
                del self._pending_commits[handle]
                future.result()
        try:
            handle = self._submit_gradients(expected, grads)
            self._gradient_bytes = grads.numel() * grads.element_size()
            self._commit = self._worker.submit(self._wait_commit, handle)
            self._pending_commits[handle] = self._commit
            self._submitted += 1
            self._queue_next_read(self._commit)
        except BaseException as error:
            self._submission_error = error
            raise
        if self._updated_ids is not None:
            self._updated_ids.update(expected.tolist())
        self._current_rows = None
        self._owner_metadata.pop(self._current_step, None)

    def _track_commit(self, handle):
        # Native acceptance precedes Python scheduling, which can still fail.
        self._pending_commits[handle] = None
        self._accepted_step = self._submitted
        return handle

    def _submit_gradients(self, ids, grads):
        return self._track_commit(self.kv_client.sparse_gradient_commit_async(
            self.master_table_name, ids, grads, session=self._session,
            step=self._submitted, rank=self._rank, world=self._world,
            lr=self.lr, divisor=self._world))

    def _close_commit_session(self, abort):
        self.kv_client.sparse_gradient_commit_close(
            self.master_table_name, session=self._session,
            step=self._accepted_step, rank=self._rank, world=self._world,
            dim=self.embedding_dim, lr=self.lr, divisor=self._world, abort=abort)

    def _finish_commits(self, *, abort):
        """Close the session and consume every accepted native handle once."""
        error = None
        if self._accepted_step >= 0 and not self._commit_session_closed:
            try:
                self._close_commit_session(abort)
            except BaseException as exc:
                # Native abort can acknowledge failure by throwing.
                if not abort:
                    error = exc
            finally:
                self._commit_session_closed = True
        for handle, future in list(self._pending_commits.items()):
            try:
                if future is None or future.cancelled():
                    self._wait_commit(handle)
                else:
                    future.result()
            except BaseException as exc:
                if error is None:
                    error = exc
                else:
                    if hasattr(error, "add_note"):
                        error.add_note("Commit drain also failed: " + repr(exc))
            finally:
                del self._pending_commits[handle]
        if error is not None:
            raise error

    @property
    def prepared_steps(self):
        return self._prepared

    @property
    def table_types(self):
        return self._owner_table_types

    def finish_training(self):
        """Include the final commit in the measured training interval."""
        self._wait_pending()

    def _wait_commit(self, handle):
        start = time.perf_counter()
        self.kv_client.sparse_gradient_commit_wait(handle)
        return (time.perf_counter() - start) * 1e3

    def _wait_pending(self):
        if self._commit is not None:
            self._commit.result()

    def validation_ids(self):
        if self._updated_ids is None:
            raise RuntimeError("PS pipeline validation tracking was not enabled")
        self._wait_pending()
        touched = sorted(self._updated_ids)
        if self._world > 1:
            ranks = [None] * self._world
            dist.all_gather_object(ranks, touched)
            touched = sorted(set(key for keys in ranks for key in keys))
        return torch.tensor(touched, dtype=torch.int64)

    def after_sparse_update(self, step, sparse_features, sparse_optimizer, row):
        del step, sparse_features, sparse_optimizer
        row["ps_pipeline_gradient_bytes"] = self._gradient_bytes

    def after_step_sync(self, row):
        del row

    def advance_all(self):
        return 0

    def shutdown(self, *, flush=True):
        if self._closed:
            return
        error = self._submission_error if flush else None
        try:
            if flush and error is None:
                self._wait_pending()
                if self._read is not None:
                    self._read.result()
        except BaseException as exc:
            error = exc
        try:
            self._finish_commits(abort=not flush or error is not None)
        except BaseException as exc:
            if error is None and flush:
                error = exc
            elif error is not None:
                if hasattr(error, "add_note"):
                    error.add_note("Commit cleanup also failed: " + repr(exc))
        finally:
            self._worker.shutdown(wait=True, cancel_futures=True)
            if self._metadata_worker is not None:
                self._metadata_worker.shutdown(wait=True, cancel_futures=True)
            if self._metadata_group is not None:
                try:
                    dist.destroy_process_group(self._metadata_group)
                except BaseException as exc:
                    if error is None and flush:
                        error = exc
                self._metadata_group = None
            self._metadata.clear()
            self._owner_metadata.clear()
            self._current_metrics = None
            self._split_clean = self._consumed_read = None
            self._closed = True
            self._read = self._commit = self._current_rows = self._current_ticket = None
            self._pending_read_ids = None
            self._current_inverse_cpu = self._current_inverse = None
        if error is not None:
            raise error


class PSPipelineSGD:
    """Coalesce pooled gradients using the prepared CPU ID ticket."""

    def __init__(self, module, pipeline):
        self.module = module
        self.pipeline = pipeline
        self.param_groups = [{"params": [module], "lr": pipeline.lr}]
        self._grads = None
        self._early_submitted = False

    def zero_grad(self):
        self.module.reset_trace()
        self._grads = None

    def record_pooled_grad(self, grad):
        if self._grads is not None or self._early_submitted:
            raise RuntimeError("PS pipeline gradient already recorded or submitted")
        ids, _, raw_count = self.pipeline._current_ticket
        features = len(self.pipeline._keys)
        if (grad.ndim != 3 or grad.shape[1:] != (features, self.pipeline.embedding_dim)
                or grad.size(0) * features != raw_count or grad.dtype != torch.float32
                or grad.device != self.pipeline._current_inverse.device):
            raise ValueError("PS pipeline pooled gradient does not match the prepared batch")
        rows = self.pipeline._current_inverse
        self._grads = torch.zeros((ids.numel(), grad.size(2)),
                                  dtype=grad.dtype, device=grad.device)
        # Keep repeated-key summation reproducible without changing dense kernels.
        previous_mode = torch.get_deterministic_debug_mode()
        try:
            torch.set_deterministic_debug_mode(2)
            self._grads.index_add_(0, rows, grad.reshape(-1, grad.size(2)))
        finally:
            torch.set_deterministic_debug_mode(previous_mode)

    def _submit(self):
        ids = self.pipeline._current_ticket[0]
        if self._grads is None:
            if ids.numel():
                raise RuntimeError("PS pipeline sparse gradient is missing")
            self._grads = torch.empty((0, self.pipeline.embedding_dim), device=self.pipeline.device)
        self.pipeline.submit_sparse_grad(ids, self._grads)
        self._grads = None

    def submit_pooled_grad(self, grad):
        """Autograd hook: enqueue event-ordered D2H/commit, preserving grad."""
        self.record_pooled_grad(grad)
        self._submit()
        self._early_submitted = True
        return grad

    def step(self):
        if self._early_submitted:
            self._early_submitted = False
        else:
            self._submit()

    def flush(self):
        # Consumption of the next batch waits for this commit in the worker.
        pass
