"""Self-check for PrefetchReadPath trainer-clock issue semantics."""

from __future__ import annotations

import unittest
from collections import deque

import torch

from ..embedding_read_path import PrefetchReadPath


class _FakeEmbeddingModule:
    """Records issue/attach calls; each handle maps back to its features."""

    def __init__(self) -> None:
        self._enable_fusion = True
        self.issued: list[tuple[object, bool]] = []
        self.attached_handles: list[int] = []

    def issue_fused_prefetch(self, features, *, record_handle: bool = True):
        self.issued.append((features, record_handle))
        handle = len(self.issued)
        return (
            handle,
            3,
            1.0,
            torch.tensor([handle], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    def set_fused_prefetch_handle(self, handle, **kwargs) -> None:
        del kwargs
        self.attached_handles.append(int(handle))

    def features_for_handle(self, handle: int):
        return self.issued[handle - 1][0]


class TestPrefetchReadPathLookahead(unittest.TestCase):
    def _run_pipeline(self, total_steps: int, depth: int):
        module = _FakeEmbeddingModule()
        path = PrefetchReadPath(module, prefetch_depth=depth)
        features = {step: object() for step in range(total_steps)}
        observed_depth = path.depth * 2  # mirror the runner's prepare window
        prepared: deque[int] = deque()
        bootstrap_issued: list[list[object]] = []
        update_issued: list[list[object]] = []
        attached_features: list[object] = []

        for step in range(total_steps):
            while (
                len(prepared) <= observed_depth
                and step + len(prepared) < total_steps
            ):
                batch_step = step + len(prepared)
                path.on_batch_prepared(batch_step, features[batch_step], None, {})
                prepared.append(batch_step)
            if step + len(prepared) >= total_steps:
                path.advance_all()

            current = prepared.popleft()
            before = len(module.issued)
            path.before_lookup(current, features[current], None, {})
            bootstrap_issued.append(
                [feat for feat, _ in module.issued[before:]]
            )
            attached_features.append(
                module.features_for_handle(module.attached_handles[-1])
            )

            # The runner prepares future batches inside the sparse-update
            # window before calling after_sparse_update.
            while (
                len(prepared) <= observed_depth
                and step + 1 + len(prepared) < total_steps
            ):
                batch_step = step + 1 + len(prepared)
                path.on_batch_prepared(batch_step, features[batch_step], None, {})
                prepared.append(batch_step)

            before = len(module.issued)
            path.after_sparse_update(current, features[current], None, {})
            update_issued.append([feat for feat, _ in module.issued[before:]])
        return module, features, bootstrap_issued, update_issued, attached_features

    def test_issues_read_depth_steps_ahead_on_trainer_clock(self) -> None:
        total_steps, depth = 10, 2
        module, features, bootstrap, update, attached = self._run_pipeline(
            total_steps, depth
        )

        # Every issue goes through record_handle=False.
        self.assertTrue(all(flag is False for _, flag in module.issued))
        self.assertEqual(len(module.issued), total_steps)

        # Only step 0 bootstraps at its own lookup; nothing else does.
        self.assertEqual(bootstrap[0], [features[0]])
        self.assertTrue(all(not issued for issued in bootstrap[1:]))

        # Step 0's update issues the first `depth` future batches; step i
        # issues the batch for step i + depth exactly.
        self.assertEqual(update[0], [features[1], features[2]])
        for step in range(1, 5):
            self.assertEqual(update[step], [features[step + depth]])
        # Once the prepare window reaches the final step, advance_all drains
        # the rest, so the trailing updates issue nothing.
        self.assertTrue(all(not issued for issued in update[5:]))

        # Each step attaches the handle issued for its own features.
        self.assertEqual(attached, [features[step] for step in range(total_steps)])

    def test_missing_slot_raises_loudly(self) -> None:
        module = _FakeEmbeddingModule()
        path = PrefetchReadPath(module, prefetch_depth=1)
        path.on_batch_prepared(0, object(), None, {})
        with self.assertRaisesRegex(RuntimeError, "prefetch slot missing for step 1"):
            path.before_lookup(1, object(), None, {})

    def test_depth_zero_stays_same_step(self) -> None:
        module = _FakeEmbeddingModule()
        path = PrefetchReadPath(module, prefetch_depth=0)
        features = object()
        ticket = path.on_batch_prepared(0, features, None, {})
        self.assertEqual(ticket, "issue_on_lookup")
        self.assertEqual(module.issued, [])
        path.before_lookup(0, features, ticket, {})
        self.assertEqual(module.issued, [(features, True)])
        path.after_sparse_update(0, features, None, {})
        self.assertEqual(len(module.issued), 1)


if __name__ == "__main__":
    unittest.main()
