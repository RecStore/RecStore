import unittest

import torch

from ..bagpipe_cache import BagPipeCachePolicy


class TestBagPipeCachePolicy(unittest.TestCase):
    def test_lookahead_ttl_and_prefetch_ids_follow_future_last_use(self):
        policy = BagPipeCachePolicy(lookahead_depth=3, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([1, 2], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([2, 3], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([3, 4], dtype=torch.int64))
        policy.observe_batch(3, torch.tensor([2, 5], dtype=torch.int64))

        plan = policy.plan_for_step(0)

        self.assertEqual(plan.step, 0)
        self.assertEqual(plan.prefetch_ids.tolist(), [1, 2])
        self.assertEqual(plan.cache_insert_ids.tolist(), [2])
        self.assertEqual(plan.ttl_updates, {2: 3})
        self.assertEqual(plan.sync_now_ids.tolist(), [2])
        self.assertEqual(plan.sync_later_ids.tolist(), [])
        self.assertEqual(plan.no_sync_ids.tolist(), [1])

    def test_no_sync_ids_are_pruned_from_sync_later(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([1, 2, 3], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([2], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([3], dtype=torch.int64))

        plan = policy.plan_for_step(0)

        self.assertEqual(plan.sync_now_ids.tolist(), [2])
        self.assertEqual(plan.no_sync_ids.tolist(), [1])
        self.assertEqual(plan.sync_later_ids.tolist(), [3])

    def test_update_before_prefetch_issue_does_not_mark_future_consumption_stale(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([1], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([7], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([7, 8], dtype=torch.int64))

        policy.plan_for_step(0)
        policy.plan_for_step(1)
        policy.plan_for_step(2)

        policy.on_update(step=1, ids=torch.tensor([7], dtype=torch.int64))
        decision = policy.on_consume(step=2)

        self.assertEqual(decision.stale_ids.tolist(), [])
        self.assertEqual(decision.valid_prefetch_ids.tolist(), [])
        self.assertFalse(decision.requires_refetch)

    def test_update_after_prefetch_issue_marks_future_consumption_stale(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([1], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([7], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([7, 8], dtype=torch.int64))

        policy.plan_for_step(0)
        policy.plan_for_step(1)
        policy.plan_for_step(2)
        policy.on_prefetch_issued(2, torch.tensor([7, 8], dtype=torch.int64))

        policy.on_update(step=1, ids=torch.tensor([7], dtype=torch.int64))
        decision = policy.on_consume(step=2)

        self.assertEqual(decision.stale_ids.tolist(), [7])
        self.assertEqual(decision.valid_prefetch_ids.tolist(), [8])
        self.assertTrue(decision.requires_refetch)

    def test_prefetch_issue_after_update_clears_prior_stale_marker(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([1], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([7], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([7, 8], dtype=torch.int64))

        policy.plan_for_step(0)
        policy.plan_for_step(1)
        policy.plan_for_step(2)
        policy.on_update(step=1, ids=torch.tensor([7], dtype=torch.int64))
        policy.on_prefetch_issued(2, torch.tensor([7, 8], dtype=torch.int64))

        decision = policy.on_consume(step=2)

        self.assertEqual(decision.stale_ids.tolist(), [])
        self.assertEqual(decision.valid_prefetch_ids.tolist(), [7, 8])
        self.assertFalse(decision.requires_refetch)

    def test_eviction_expires_ids_after_ttl(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([5], dtype=torch.int64))

        policy.plan_for_step(0)
        self.assertEqual(policy.cached_ids(), {4})

        policy.on_step_end(0)
        self.assertEqual(policy.cached_ids(), {4})

        policy.on_step_end(1)
        self.assertEqual(policy.cached_ids(), set())

        evicted = policy.on_step_end(2)
        self.assertEqual(evicted.tolist(), [])
        self.assertEqual(policy.cached_ids(), set())

    def test_cached_future_id_is_not_network_prefetched_again(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([4, 5], dtype=torch.int64))

        first = policy.plan_for_step(0)
        second = policy.plan_for_step(1)

        self.assertEqual(first.prefetch_ids.tolist(), [4])
        self.assertEqual(first.cache_insert_ids.tolist(), [4])
        self.assertEqual(second.prefetch_ids.tolist(), [])
        self.assertEqual(second.cache_insert_ids.tolist(), [])

    def test_capacity_overflow_raises_instead_of_evicing_live_rows(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=1)
        policy.observe_batch(0, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([4, 5], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([5], dtype=torch.int64))

        first = policy.plan_for_step(0)
        with self.assertRaisesRegex(RuntimeError, "BagPipe cache capacity exceeded"):
            policy.plan_for_step(1)

        self.assertEqual(first.evict_ids.tolist(), [])
        self.assertEqual(policy.cached_ids(), {4, 5})

    def test_cached_hot_id_gets_ttl_update_when_window_slides(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        for step in range(4):
            policy.observe_batch(step, torch.tensor([4], dtype=torch.int64))

        first = policy.plan_for_step(0)
        policy.on_step_end(0)
        second = policy.plan_for_step(1)
        policy.on_step_end(1)
        third = policy.plan_for_step(2)
        policy.on_step_end(2)

        self.assertEqual(first.ttl_updates, {4: 2})
        self.assertEqual(second.prefetch_ids.tolist(), [])
        self.assertEqual(second.ttl_updates, {4: 3})
        self.assertEqual(third.prefetch_ids.tolist(), [])
        self.assertEqual(policy.cached_ids(), {4})

    def test_gpu_cache_update_keeps_cached_id_but_does_not_require_refetch(self):
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([5], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([4], dtype=torch.int64))

        policy.plan_for_step(0)
        policy.plan_for_step(1)
        policy.plan_for_step(2)
        self.assertEqual(policy.cached_ids(), {4})

        policy.on_update(
            step=0,
            ids=torch.tensor([4], dtype=torch.int64),
            cache_updated_ids=torch.tensor([4], dtype=torch.int64),
        )

        self.assertEqual(policy.cached_ids(), {4})
        decision = policy.on_consume(2)
        self.assertEqual(decision.stale_ids.tolist(), [])
        self.assertEqual(decision.stale_cached_ids.tolist(), [])
        self.assertEqual(decision.stale_refetch_ids.tolist(), [])
        self.assertFalse(decision.requires_refetch)

    def test_step_end_keeps_rows_whose_ttl_is_next_step(self):
        policy = BagPipeCachePolicy(lookahead_depth=1, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([4], dtype=torch.int64))

        policy.plan_for_step(0)
        policy.plan_for_step(1)
        policy.on_update(
            step=0,
            ids=torch.tensor([4], dtype=torch.int64),
            cache_updated_ids=torch.tensor([4], dtype=torch.int64),
        )
        evicted = policy.on_step_end(0)

        self.assertEqual(evicted.tolist(), [])
        decision = policy.on_consume(1)
        self.assertEqual(decision.stale_ids.tolist(), [])
        self.assertEqual(decision.stale_cached_ids.tolist(), [])
        self.assertEqual(decision.stale_refetch_ids.tolist(), [])

    def test_future_planning_does_not_evict_current_cached_rows(self):
        policy = BagPipeCachePolicy(lookahead_depth=1, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([4, 5], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([5], dtype=torch.int64))

        policy.plan_for_step(0)
        policy.plan_for_step(1)
        policy.on_update(
            step=0,
            ids=torch.tensor([4], dtype=torch.int64),
            cache_updated_ids=torch.tensor([4], dtype=torch.int64),
        )
        policy.plan_for_step(2)

        decision = policy.on_consume(1)
        self.assertEqual(decision.stale_ids.tolist(), [])
        self.assertEqual(decision.stale_cached_ids.tolist(), [])
        self.assertEqual(decision.stale_refetch_ids.tolist(), [])

    def test_algorithm_one_cache_state_expires_at_each_planned_step(self):
        policy = BagPipeCachePolicy(lookahead_depth=3, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([1, 2], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([2, 3], dtype=torch.int64))
        policy.observe_batch(2, torch.tensor([3, 4], dtype=torch.int64))
        policy.observe_batch(3, torch.tensor([2, 5], dtype=torch.int64))

        expected = [
            ([1, 2], [2], {2}),
            ([3], [3], {2, 3}),
            ([4], [], {2}),
            ([5], [], set()),
        ]
        for step, (prefetch_ids, cache_insert_ids, cached_after) in enumerate(expected):
            plan = policy.plan_for_step(step)
            self.assertEqual(plan.prefetch_ids.tolist(), prefetch_ids)
            self.assertEqual(plan.cache_insert_ids.tolist(), cache_insert_ids)
            policy.on_step_end(step)
            self.assertEqual(policy.cached_ids(), cached_after)

    def test_previous_cached_update_does_not_make_current_prefetch_stale(self):
        policy = BagPipeCachePolicy(lookahead_depth=1, cache_capacity=8)
        policy.observe_batch(0, torch.tensor([4], dtype=torch.int64))
        policy.observe_batch(1, torch.tensor([4], dtype=torch.int64))

        policy.plan_for_step(0)
        policy.on_update(
            step=0,
            ids=torch.tensor([4], dtype=torch.int64),
            cache_updated_ids=torch.tensor([4], dtype=torch.int64),
        )
        policy.on_step_end(0)
        policy.plan_for_step(1)

        decision = policy.on_consume(1)
        self.assertEqual(decision.stale_ids.tolist(), [])
        self.assertEqual(decision.valid_prefetch_ids.tolist(), [])


if __name__ == "__main__":
    unittest.main()
