"""Test suite for lru_cache.py. Standard library only -- no pytest required.

Run:
    python3 -m unittest -v
    python3 -m unittest test_lru_cache.TestSpecExample -v
    python3 test_lru_cache.py
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
import unittest
from collections import OrderedDict

from lru_cache import MISS, Cache, CacheStats, FakeClock, LRUCache

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(ROOT, "demo.py")

# A unique object, so "absent" is distinguishable from any storable value.
SENTINEL = object()


def assert_invariants(tc: unittest.TestCase, cache: LRUCache) -> None:
    """Walk the chain both ways and cross-check it against the hash map."""
    head, tail = cache._head, cache._tail

    tc.assertIsNone(head.prev, "head sentinel must have no prev")
    tc.assertIsNone(tail.next, "tail sentinel must have no next")
    tc.assertIsNone(head.key)
    tc.assertIsNone(tail.key)

    forward = []
    node = head.next
    steps = 0
    while node is not tail:
        tc.assertIsNotNone(node, "chain terminated before reaching the tail")
        tc.assertIsNot(node.next, node, "node points at itself")
        forward.append(node)
        node = node.next
        steps += 1
        tc.assertLessEqual(steps, cache.capacity + 1, "forward walk did not terminate")

    backward = []
    node = tail.prev
    steps = 0
    while node is not head:
        backward.append(node)
        node = node.prev
        steps += 1
        tc.assertLessEqual(steps, cache.capacity + 1, "backward walk did not terminate")

    tc.assertEqual(
        [id(n) for n in forward],
        [id(n) for n in reversed(backward)],
        "forward and backward traversals disagree",
    )
    tc.assertEqual(len(forward), len(cache._map), "map size != linked node count")
    tc.assertLessEqual(len(cache._map), cache.capacity, "cache exceeded capacity")

    for n in forward:
        tc.assertIn(n.key, cache._map, "linked node missing from the map")
        tc.assertIs(cache._map[n.key], n, "map entry points at a different node")
    for key, n in cache._map.items():
        tc.assertIn(id(n), [id(f) for f in forward], "mapped node is unreachable")
        tc.assertEqual(key, n.key, "node stored under the wrong key")

    if not cache._map:
        tc.assertIs(head.next, tail, "empty cache must have head.next is tail")

    snapshot = cache.stats
    tc.assertEqual(snapshot.size, len(forward), "stats.size disagrees with the chain")
    tc.assertEqual(snapshot.capacity, cache.capacity, "stats.capacity is wrong")
    for field in ("hits", "misses", "evictions", "expirations"):
        tc.assertGreaterEqual(getattr(snapshot, field), 0, "%s went negative" % field)


# ----------------------------------------------------------------------
class TestConstruction(unittest.TestCase):
    def test_rejects_zero_capacity(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            Cache(0)

    def test_rejects_negative_capacity(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            Cache(-3)

    def test_rejects_float_capacity(self):
        with self.assertRaisesRegex(TypeError, "float"):
            Cache(2.0)

    def test_rejects_bool_capacity(self):
        # bool subclasses int; Cache(True) silently meaning capacity 1 is a bug.
        with self.assertRaisesRegex(TypeError, "bool"):
            Cache(True)

    def test_rejects_none_capacity(self):
        with self.assertRaises(TypeError):
            Cache(None)

    def test_rejects_str_capacity(self):
        with self.assertRaises(TypeError):
            Cache("2")

    def test_rejects_non_callable_time_fn(self):
        with self.assertRaisesRegex(TypeError, "time_fn"):
            Cache(2, time_fn=123)

    def test_capacity_one_is_valid(self):
        cache = Cache(1)
        self.assertEqual(cache.capacity, 1)
        self.assertEqual(len(cache), 0)

    def test_initial_state(self):
        cache = Cache(4)
        self.assertEqual(cache.keys(), [])
        self.assertEqual(cache.stats.hits, 0)
        self.assertEqual(cache.stats.hit_rate, 0.0)
        self.assertIsInstance(cache.stats, CacheStats)
        repr(cache)  # must not raise
        assert_invariants(self, cache)

    def test_capacity_is_read_only(self):
        cache = Cache(2)
        with self.assertRaises(AttributeError):
            cache.capacity = 5

    def test_default_time_fn_is_monotonic(self):
        self.assertIs(Cache(2).time_fn, time.monotonic)

    def test_rejects_invalid_default_ttl(self):
        with self.assertRaises(ValueError):
            Cache(2, default_ttl=0)
        with self.assertRaises(TypeError):
            Cache(2, default_ttl="5")


# ----------------------------------------------------------------------
class TestSpecExample(unittest.TestCase):
    """The exact sequence from the assignment brief."""

    def test_assignment_canonical_example(self):
        cache = Cache(2)
        cache.put("A", 10)
        cache.put("B", 20)
        self.assertEqual(cache.get("A"), 10)
        cache.put("C", 30)
        self.assertEqual(cache.get("B"), -1)
        self.assertEqual(cache.get("C"), 30)
        self.assertEqual(cache.get("A"), 10)

        self.assertEqual(cache.keys(), ["A", "C"])
        self.assertEqual(cache.stats.evictions, 1)
        assert_invariants(self, cache)


# ----------------------------------------------------------------------
class TestBasicOperations(unittest.TestCase):
    def test_get_missing_returns_minus_one(self):
        self.assertEqual(Cache(2).get("nope"), -1)

    def test_get_missing_returns_custom_default(self):
        self.assertIs(Cache(2).get("nope", SENTINEL), SENTINEL)

    def test_put_then_get_roundtrip(self):
        cache = Cache(2)
        cache.put("k", "v")
        self.assertEqual(cache.get("k"), "v")

    def test_put_existing_replaces_value_without_growth(self):
        cache = Cache(2)
        cache.put("k", 1)
        cache.put("k", 2)
        self.assertEqual(cache.get("k"), 2)
        self.assertEqual(len(cache), 1)

    def test_put_existing_at_capacity_does_not_evict(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        for i in range(3):
            cache.put("A", i)
        self.assertEqual(cache.stats.evictions, 0)
        self.assertIn("A", cache)
        self.assertIn("B", cache)
        assert_invariants(self, cache)

    def test_none_value_is_not_a_miss(self):
        cache = Cache(2)
        cache.put("k", None)
        self.assertIsNone(cache.get("k"))
        self.assertIn("k", cache)

    def test_value_of_minus_one_is_distinguishable_via_sentinel(self):
        cache = Cache(2)
        cache.put("k", -1)
        # The documented ambiguity: with the spec default, a stored -1 looks
        # exactly like a miss.
        self.assertEqual(cache.get("k"), -1)
        self.assertEqual(cache.get("absent"), -1)
        # The documented workarounds.
        self.assertIs(cache.get("absent", SENTINEL), SENTINEL)
        self.assertEqual(cache.get("k", SENTINEL), -1)
        self.assertIn("k", cache)
        self.assertNotIn("absent", cache)

    def test_key_of_minus_one_works(self):
        cache = Cache(2)
        cache.put(-1, "negative key")
        self.assertEqual(cache.get(-1), "negative key")

    def test_distinct_key_types_do_not_collide(self):
        cache = Cache(4)
        cache.put(1, "int")
        cache.put("1", "str")
        cache.put((1, 2), "tuple")
        self.assertEqual(cache.get(1), "int")
        self.assertEqual(cache.get("1"), "str")
        self.assertEqual(cache.get((1, 2)), "tuple")
        self.assertEqual(len(cache), 3)

    def test_unhashable_key_raises_on_put(self):
        with self.assertRaises(TypeError):
            Cache(2).put([1, 2], "v")

    def test_unhashable_key_raises_on_get(self):
        with self.assertRaises(TypeError):
            Cache(2).get([1, 2])

    def test_unhashable_key_leaves_cache_untouched(self):
        cache = Cache(2)
        cache.put("A", 1)
        with self.assertRaises(TypeError):
            cache.put({"unhashable"}, 2)
        self.assertEqual(len(cache), 1)
        assert_invariants(self, cache)

    def test_len_and_iteration(self):
        cache = Cache(3)
        for key in "ABC":
            cache.put(key, key.lower())
        self.assertEqual(len(cache), 3)
        self.assertEqual(list(cache), ["C", "B", "A"])
        self.assertEqual(cache.items(), [("C", "c"), ("B", "b"), ("A", "a")])

    def test_clear_releases_values_immediately(self):
        # clear() must unlink as it goes. Leaving the discarded nodes pointing
        # at each other would be a reference cycle, keeping every cached value
        # alive until the cyclic collector happens to run.
        import gc
        import weakref

        class Holder:
            pass

        cache = Cache(200)
        refs = []
        gc.disable()
        try:
            for i in range(200):
                value = Holder()
                refs.append(weakref.ref(value))
                cache.put(i, value)
            del value
            cache.clear()
            alive = sum(1 for ref in refs if ref() is not None)
        finally:
            gc.enable()
        self.assertEqual(alive, 0, "clear() leaked %d values into a cycle" % alive)

    def test_clear_empties_cache_and_keeps_it_usable(self):
        cache = Cache(3)
        for key in "ABC":
            cache.put(key, 1)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.keys(), [])
        assert_invariants(self, cache)
        cache.put("D", 4)
        self.assertEqual(cache.get("D"), 4)
        assert_invariants(self, cache)

    def test_repr_contains_no_memory_address(self):
        cache = Cache(2)
        cache.put("A", 1)
        text = repr(cache)
        self.assertNotIn("0x", text)
        self.assertIn("capacity=2", text)

    def test_repr_truncates_long_key_lists(self):
        cache = Cache(20)
        for i in range(20):
            cache.put(i, i)
        self.assertIn("+12 more", repr(cache))


# ----------------------------------------------------------------------
class TestEvictionAndRecency(unittest.TestCase):
    def test_evicts_least_recently_used_when_full(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.put("C", 3)
        self.assertNotIn("A", cache)
        self.assertEqual(cache.keys(), ["C", "B"])

    def test_successful_get_refreshes_recency(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.get("A")  # A becomes MRU, so B is now the eviction victim
        cache.put("C", 3)
        self.assertIn("A", cache)
        self.assertNotIn("B", cache)

    def test_put_update_refreshes_recency(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.put("A", 99)
        cache.put("C", 3)
        self.assertIn("A", cache)
        self.assertNotIn("B", cache)

    def test_failed_get_does_not_reorder(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.get("Z")  # miss -- must be completely inert
        cache.put("C", 3)
        self.assertNotIn("A", cache, "a miss must not have promoted anything")
        self.assertIn("B", cache)

    def test_peek_does_not_change_recency(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        self.assertEqual(cache.peek("A"), 1)
        cache.put("C", 3)
        self.assertNotIn("A", cache)

    def test_contains_does_not_change_recency(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        self.assertIn("A", cache)
        cache.put("C", 3)
        self.assertNotIn("A", cache)

    def test_keys_does_not_change_recency(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.keys()
        cache.put("C", 3)
        self.assertNotIn("A", cache)

    def test_peek_missing_returns_default(self):
        self.assertEqual(Cache(2).peek("nope"), -1)
        self.assertIs(Cache(2).peek("nope", SENTINEL), SENTINEL)

    def test_capacity_one_evicts_on_every_new_key(self):
        cache = Cache(1)
        for i in range(5):
            cache.put(i, i)
            self.assertEqual(len(cache), 1)
            assert_invariants(self, cache)
        self.assertEqual(cache.keys(), [4])
        self.assertEqual(cache.stats.evictions, 4)

    def test_keys_are_ordered_mru_to_lru(self):
        cache = Cache(3)
        cache.put("A", 1)
        cache.put("B", 2)
        cache.put("C", 3)
        self.assertEqual(cache.keys(), ["C", "B", "A"])
        cache.get("A")
        self.assertEqual(cache.keys(), ["A", "C", "B"])

    def test_eviction_counter_tracks_evictions(self):
        cache = Cache(2)
        for i in range(10):
            cache.put(i, i)
        self.assertEqual(cache.stats.evictions, 8)

    def test_never_exceeds_capacity_under_sustained_inserts(self):
        cache = Cache(10)
        for i in range(1000):
            cache.put(i, i)
            self.assertEqual(len(cache), min(i + 1, 10))
        self.assertEqual(cache.keys(), list(range(999, 989, -1)))
        assert_invariants(self, cache)

    def test_invariants_hold_after_mixed_operations(self):
        cache = Cache(5)
        rng = random.Random(1234)
        for i in range(200):
            if rng.random() < 0.5:
                cache.put(rng.randrange(12), i)
            else:
                cache.get(rng.randrange(12))
            assert_invariants(self, cache)


# ----------------------------------------------------------------------
class TestPurity(unittest.TestCase):
    """Read-only operations must change neither the ordering nor the counters."""

    def test_read_only_operations_do_not_mutate_state(self):
        clock = FakeClock(0.0)
        cache = Cache(4, time_fn=clock)
        cache.put("live", 1)
        cache.put("dead", 2, ttl=1.0)
        cache.put("other", 3)
        clock.set(5.0)

        before_stats = cache.stats
        before_order = cache.keys()
        before_len = len(cache)

        cache.peek("live")
        cache.peek("dead")
        cache.peek("absent")
        self.assertIn("live", cache)
        self.assertNotIn("dead", cache)
        self.assertNotIn("absent", cache)
        cache.keys()
        cache.items()
        list(cache)
        repr(cache)
        len(cache)

        self.assertEqual(cache.stats, before_stats, "a pure operation moved a counter")
        self.assertEqual(cache.keys(), before_order, "a pure operation reordered")
        self.assertEqual(len(cache), before_len, "a pure operation reclaimed a node")


class TestPublicSurface(unittest.TestCase):
    def test_cache_is_an_alias_for_lru_cache(self):
        self.assertIs(Cache, LRUCache)

    def test_miss_constant_is_the_spec_sentinel(self):
        self.assertEqual(MISS, -1)
        self.assertEqual(Cache(2).get("absent"), MISS)

    def test_default_ttl_property_reports_the_configured_value(self):
        self.assertIsNone(Cache(2).default_ttl)
        self.assertEqual(Cache(2, default_ttl=5).default_ttl, 5.0)

    def test_stats_reports_size_and_capacity(self):
        cache = Cache(3)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.stats.size, 2)
        self.assertEqual(cache.stats.capacity, 3)

    def test_items_and_iter_agree_with_keys(self):
        cache = Cache(3)
        for key in "abc":
            cache.put(key, key.upper())
        self.assertEqual([k for k, _ in cache.items()], cache.keys())
        self.assertEqual(list(cache), cache.keys())


class TestStats(unittest.TestCase):
    def test_hits_and_misses_counted(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.get("A")
        cache.get("A")
        cache.get("Z")
        self.assertEqual(cache.stats.hits, 2)
        self.assertEqual(cache.stats.misses, 1)

    def test_hit_rate_is_zero_with_no_lookups(self):
        self.assertEqual(Cache(2).stats.hit_rate, 0.0)  # not ZeroDivisionError

    def test_hit_rate_computes_correctly(self):
        cache = Cache(2)
        cache.put("A", 1)
        cache.get("A")
        cache.get("A")
        cache.get("A")
        cache.get("Z")
        self.assertAlmostEqual(cache.stats.hit_rate, 0.75)

    def test_clear_preserves_all_four_lifetime_counters(self):
        clock = FakeClock(0.0)
        cache = Cache(2, time_fn=clock)
        cache.put("A", 1, ttl=1.0)
        cache.get("A")                 # hit
        cache.get("Z")                 # miss
        clock.set(5.0)
        cache.get("A")                 # miss + expiration
        cache.put("B", 2)
        cache.put("C", 3)
        cache.put("D", 4)              # forces an eviction
        before = cache.stats
        self.assertGreater(before.evictions, 0)
        self.assertGreater(before.expirations, 0)
        cache.clear()
        after = cache.stats
        for field in ("hits", "misses", "evictions", "expirations"):
            self.assertEqual(getattr(after, field), getattr(before, field),
                             "clear() reset %s" % field)
        self.assertEqual(after.size, 0)


# ----------------------------------------------------------------------
class TestTTL(unittest.TestCase):
    """Every case uses FakeClock: no sleeps, no flakiness."""

    def setUp(self):
        self.clock = FakeClock(0.0)

    def make(self, capacity=3, default_ttl=None):
        return Cache(capacity, default_ttl=default_ttl, time_fn=self.clock)

    def test_entry_is_alive_before_expiry(self):
        cache = self.make()
        cache.put("k", "v", ttl=10)
        self.clock.advance(9.99)
        self.assertEqual(cache.get("k"), "v")

    def test_entry_expires_exactly_at_boundary(self):
        # Pins the ">=" choice: validity is the half-open window [t0, t0+ttl).
        cache = self.make()
        cache.put("k", "v", ttl=5)
        self.clock.set(5.0)
        self.assertEqual(cache.get("k"), -1)

    def test_entry_expires_after_boundary(self):
        cache = self.make()
        cache.put("k", "v", ttl=5)
        self.clock.set(5.001)
        self.assertEqual(cache.get("k"), -1)
        self.assertEqual(cache.stats.expirations, 1)

    def test_default_ttl_applies_to_every_put(self):
        cache = self.make(default_ttl=2.0)
        cache.put("a", 1)
        cache.put("b", 2)
        self.clock.advance(2.0)
        self.assertEqual(cache.get("a"), -1)
        self.assertEqual(cache.get("b"), -1)

    def test_per_put_ttl_overrides_default_ttl(self):
        cache = self.make(default_ttl=2.0)
        cache.put("short", 1)
        cache.put("long", 2, ttl=100.0)
        self.clock.advance(3.0)
        self.assertEqual(cache.get("short"), -1)
        self.assertEqual(cache.get("long"), 2)

    def test_explicit_none_ttl_disables_default_ttl(self):
        cache = self.make(default_ttl=2.0)
        cache.put("forever", 1, ttl=None)
        self.clock.advance(10_000.0)
        self.assertEqual(cache.get("forever"), 1)

    def test_entries_without_ttl_never_expire(self):
        cache = self.make()
        cache.put("k", "v")
        self.clock.advance(1e9)
        self.assertEqual(cache.get("k"), "v")

    def test_rejects_zero_and_negative_ttl(self):
        cache = self.make()
        with self.assertRaises(ValueError):
            cache.put("k", 1, ttl=0)
        with self.assertRaises(ValueError):
            cache.put("k", 1, ttl=-5)

    def test_rejects_non_numeric_ttl(self):
        cache = self.make()
        with self.assertRaises(TypeError):
            cache.put("k", 1, ttl="10")
        with self.assertRaises(TypeError):
            cache.put("k", 1, ttl=True)

    def test_expired_entry_frees_capacity_on_access(self):
        cache = self.make(capacity=2, default_ttl=1.0)
        cache.put("a", 1)
        cache.put("b", 2)
        self.clock.advance(2.0)
        self.assertEqual(cache.get("a"), -1)
        self.assertEqual(cache.get("b"), -1)
        self.assertEqual(len(cache), 0)
        cache.put("c", 3)
        cache.put("d", 4)
        self.assertEqual(cache.stats.evictions, 0)
        self.assertEqual(cache.stats.expirations, 2)

    def test_put_on_expired_key_is_update_not_insert(self):
        cache = self.make(capacity=2, default_ttl=1.0)
        cache.put("a", 1)
        self.clock.advance(2.0)
        cache.put("a", 99)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.stats.evictions, 0)
        self.assertEqual(cache.stats.expirations, 0)
        self.assertEqual(cache.get("a"), 99)
        assert_invariants(self, cache)

    def test_contains_is_false_for_expired_but_does_not_reclaim(self):
        cache = self.make(default_ttl=1.0)
        cache.put("k", 1)
        self.clock.advance(2.0)
        self.assertNotIn("k", cache)          # liveness question -> False
        self.assertEqual(len(cache), 1)       # capacity question -> still resident
        self.assertEqual(cache.purge_expired(), 1)
        self.assertEqual(len(cache), 0)
        assert_invariants(self, cache)

    def test_peek_is_pure_for_expired_entries(self):
        cache = self.make(default_ttl=1.0)
        cache.put("k", 1)
        self.clock.advance(2.0)
        self.assertEqual(cache.peek("k"), -1)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.stats.expirations, 0)

    def test_expired_entries_omitted_from_keys_and_items(self):
        cache = self.make()
        cache.put("live", 1)
        cache.put("dead", 2, ttl=1.0)
        self.clock.advance(2.0)
        self.assertEqual(cache.keys(), ["live"])
        self.assertEqual(cache.items(), [("live", 1)])

    def test_purge_expired_returns_count_and_leaves_live_entries(self):
        cache = self.make(capacity=5)
        cache.put("a", 1, ttl=1.0)
        cache.put("b", 2, ttl=1.0)
        cache.put("c", 3)
        self.clock.advance(2.0)
        self.assertEqual(cache.purge_expired(), 2)
        self.assertEqual(cache.keys(), ["c"])
        self.assertEqual(cache.purge_expired(), 0)
        assert_invariants(self, cache)

    def test_ttl_does_not_disturb_lru_order_of_live_entries(self):
        cache = self.make(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2, ttl=1.0)
        cache.put("c", 3)
        self.clock.advance(2.0)
        cache.purge_expired()
        self.assertEqual(cache.keys(), ["c", "a"])

    def test_put_without_ttl_clears_an_existing_deadline(self):
        # put() always re-resolves the deadline, so a plain put on a key that
        # had a ttl makes it immortal. Pinning the semantics either way.
        cache = self.make()
        cache.put("k", 1, ttl=5.0)
        self.clock.advance(1.0)
        cache.put("k", 2)                   # no ttl, no default_ttl -> immortal
        self.clock.advance(10_000.0)
        self.assertEqual(cache.get("k"), 2)

    def test_put_with_default_ttl_restarts_the_clock(self):
        cache = self.make(default_ttl=5.0)
        cache.put("k", 1)
        self.clock.advance(4.0)
        cache.put("k", 2)                   # refreshes the deadline to t=9
        self.clock.advance(4.0)
        self.assertEqual(cache.get("k"), 2)
        self.clock.advance(2.0)
        self.assertEqual(cache.get("k"), -1)

    def test_expired_get_counts_a_miss_and_honours_a_custom_default(self):
        cache = self.make(default_ttl=1.0)
        cache.put("k", 1)
        self.clock.advance(2.0)
        self.assertIs(cache.get("k", SENTINEL), SENTINEL)
        self.assertEqual(cache.stats.misses, 1)
        self.assertEqual(cache.stats.expirations, 1)
        self.assertEqual(cache.stats.hit_rate, 0.0)

    def test_expired_peek_honours_a_custom_default(self):
        cache = self.make(default_ttl=1.0)
        cache.put("k", 1)
        self.clock.advance(2.0)
        self.assertIs(cache.peek("k", SENTINEL), SENTINEL)

    def test_put_on_expired_key_promotes_it_to_mru(self):
        cache = self.make(capacity=3, default_ttl=10.0)
        cache.put("a", 1, ttl=1.0)
        cache.put("b", 2)
        cache.put("c", 3)
        self.clock.advance(2.0)
        cache.put("a", 99)                  # refresh of an expired resident key
        self.assertEqual(cache.keys()[0], "a")
        assert_invariants(self, cache)

    def test_expired_get_reclaims_without_disturbing_order(self):
        # The only miss path that mutates the chain: it must unlink the dead
        # node and leave every surviving entry in the same relative order.
        cache = self.make(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2, ttl=1.0)
        cache.put("c", 3)
        self.clock.advance(2.0)
        self.assertEqual(cache.get("b"), -1)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.keys(), ["c", "a"])
        # The reclaim freed a slot, so the next insert costs no eviction.
        cache.put("d", 4)
        self.assertEqual(cache.keys(), ["d", "c", "a"])
        self.assertEqual(cache.stats.evictions, 0)
        assert_invariants(self, cache)

    def test_dead_tail_entry_is_reclaimed_rather_than_evicted(self):
        # An insert under pressure prefers reclaiming a corpse at the LRU end
        # over discarding a live entry, and attributes it to TTL not capacity.
        cache = self.make(capacity=1, default_ttl=1.0)
        cache.put("a", 1)
        self.clock.advance(2.0)
        cache.put("b", 2)
        self.assertEqual(cache.stats.evictions, 0)
        self.assertEqual(cache.stats.expirations, 1)
        self.assertEqual(cache.keys(), ["b"])
        assert_invariants(self, cache)

    def test_dead_entry_near_mru_still_holds_its_slot(self):
        # The honest limit of lazy expiry: reclamation only reaches the LRU
        # end, so a recently-touched corpse survives and a live entry is the
        # one discarded. Finding it anywhere else would cost an O(n) scan.
        cache = self.make(capacity=2)
        cache.put("dead", "x", ttl=1.0)
        cache.put("live", "y", ttl=100.0)
        cache.get("dead")            # promotes the soon-to-be corpse to MRU
        self.clock.set(2.0)
        cache.put("new", "z")
        self.assertEqual(cache.stats.evictions, 1)
        self.assertNotIn("live", cache)
        self.assertEqual(len(cache), 2)     # the corpse is still resident
        self.assertEqual(cache.keys(), ["new"])

    def test_items_classifies_every_node_against_one_clock_sample(self):
        # A walk must describe a single instant, not n different ones.
        class Tick:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                self.t += 1.0
                return self.t

        cache = Cache(4, time_fn=Tick())
        cache.put("a", 1, ttl=100.0)
        cache.put("b", 2, ttl=100.0)
        cache.put("c", 3, ttl=100.0)
        self.assertEqual(cache.keys(), ["c", "b", "a"])

    def test_lru_key_reports_the_next_victim(self):
        cache = Cache(2)
        self.assertIsNone(cache.lru_key)
        cache.put("A", 1)
        cache.put("B", 2)
        self.assertEqual(cache.lru_key, "A")
        cache.get("A")
        self.assertEqual(cache.lru_key, "B")
        cache.put("C", 3)
        self.assertNotIn("B", cache)


# ----------------------------------------------------------------------
class ReferenceLRU:
    """An independent oracle built on OrderedDict. End of the dict == MRU."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.d: "OrderedDict[object, object]" = OrderedDict()

    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]

    def peek(self, key):
        return self.d.get(key, -1)

    def put(self, key, value):
        if key in self.d:
            self.d[key] = value
            self.d.move_to_end(key)
            return
        if len(self.d) >= self.capacity:
            self.d.popitem(last=False)
        self.d[key] = value

    def order_mru_to_lru(self):
        return list(reversed(self.d))


class TestDifferentialAgainstReferenceModel(unittest.TestCase):
    """Random operations must agree with an independently written model.

    Hand-written tests find the cases you thought of. This finds the rest.
    """

    def test_reference_model_satisfies_spec_example(self):
        # Validate the oracle before trusting it, or a shared bug passes silently.
        model = ReferenceLRU(2)
        model.put("A", 10)
        model.put("B", 20)
        self.assertEqual(model.get("A"), 10)
        model.put("C", 30)
        self.assertEqual(model.get("B"), -1)
        self.assertEqual(model.get("C"), 30)
        self.assertEqual(model.get("A"), 10)
        self.assertEqual(model.order_mru_to_lru(), ["A", "C"])

    def test_random_operations_match_reference_model(self):
        for capacity in (1, 2, 3, 8, 64):
            for seed in (20260923, 7, 12345):
                with self.subTest(capacity=capacity, seed=seed):
                    self._run_differential(capacity, seed, ops=5000)

    def test_deep_random_run_matches_reference_model(self):
        self._run_differential(16, 98765, ops=50_000, order_every=100)

    def _run_differential(self, capacity, seed, ops, order_every=1):
        rng = random.Random(seed)
        cache = Cache(capacity)
        model = ReferenceLRU(capacity)
        # ReferenceLRU has no notion of TTL, so its peek() would diverge from
        # the real cache's expiry-aware peek. Assert the precondition rather
        # than relying on it by accident. TTL is covered by
        # TestDifferentialWithTTL against a TTL-aware oracle.
        self.assertIsNone(cache.default_ttl)
        key_space = 2 * capacity + 2
        history = []

        for i in range(ops):
            key = rng.randrange(key_space)
            roll = rng.random()
            if roll < 0.40:
                value = rng.randrange(1_000_000)
                history.append(("put", key, value))
                cache.put(key, value)
                model.put(key, value)
                got = expected = None
            elif roll < 0.90:
                history.append(("get", key, None))
                got = cache.get(key)
                expected = model.get(key)
            else:
                history.append(("peek", key, None))
                got = cache.peek(key)
                expected = model.peek(key)

            if got != expected:
                self.fail(self._describe(seed, capacity, i, history, got, expected))
            if i % order_every == 0 and cache.keys() != model.order_mru_to_lru():
                self.fail(
                    self._describe(
                        seed, capacity, i, history,
                        cache.keys(), model.order_mru_to_lru(), what="order",
                    )
                )
            if i % 500 == 0:
                assert_invariants(self, cache)

        self.assertEqual(cache.keys(), model.order_mru_to_lru())
        assert_invariants(self, cache)

    @staticmethod
    def _describe(seed, capacity, index, history, got, expected, what="return value"):
        return (
            "%s mismatch at op %d (seed=%d, capacity=%d)\n"
            "  cache    : %r\n  reference: %r\n  last ops : %r"
            % (what, index, seed, capacity, got, expected, history[-20:])
        )


# ----------------------------------------------------------------------
class ReferenceTTLCache:
    """Oracle for the TTL-enabled cache, including insert-time reclamation."""

    def __init__(self, capacity, clock):
        self.capacity = capacity
        self.clock = clock
        self.d = OrderedDict()          # key -> [value, expires_at]

    def _dead(self, key, now):
        exp = self.d[key][1]
        return exp is not None and now >= exp

    def get(self, key):
        now = self.clock()
        if key not in self.d:
            return -1
        if self._dead(key, now):
            del self.d[key]
            return -1
        self.d.move_to_end(key)
        return self.d[key][0]

    def peek(self, key):
        now = self.clock()
        if key not in self.d or self._dead(key, now):
            return -1
        return self.d[key][0]

    def put(self, key, value, ttl=None):
        now = self.clock()
        exp = None if ttl is None else now + ttl
        if key in self.d:
            self.d[key] = [value, exp]
            self.d.move_to_end(key)
            return
        while len(self.d) >= self.capacity:
            oldest = next(iter(self.d))
            if not self._dead(oldest, now):
                break
            del self.d[oldest]
        if len(self.d) >= self.capacity:
            self.d.popitem(last=False)
        self.d[key] = [value, exp]

    def order_mru_to_lru(self):
        now = self.clock()
        return [k for k in reversed(self.d) if not self._dead(k, now)]


class TestDifferentialWithTTL(unittest.TestCase):
    """The TTL-free differential test never exercises expiry. This one does."""

    def test_random_ttl_operations_match_reference_model(self):
        for capacity in (1, 2, 5, 16):
            for seed in (31337, 2024):
                with self.subTest(capacity=capacity, seed=seed):
                    self._run(capacity, seed, ops=4000)

    def _run(self, capacity, seed, ops):
        rng = random.Random(seed)
        clock = FakeClock(0.0)
        cache = Cache(capacity, time_fn=clock)
        model = ReferenceTTLCache(capacity, clock)
        key_space = 2 * capacity + 2
        history = []

        for i in range(ops):
            key = rng.randrange(key_space)
            roll = rng.random()
            if roll < 0.45:
                ttl = rng.choice([None, 1.0, 3.0, 10.0])
                value = rng.randrange(1_000_000)
                history.append(("put", key, value, ttl))
                cache.put(key, value, ttl=ttl)
                model.put(key, value, ttl=ttl)
                got = expected = None
            elif roll < 0.85:
                history.append(("get", key, None, None))
                got, expected = cache.get(key), model.get(key)
            else:
                history.append(("peek", key, None, None))
                got, expected = cache.peek(key), model.peek(key)

            if got != expected:
                self.fail("value mismatch at op %d (seed=%d cap=%d): %r != %r\n%r"
                          % (i, seed, capacity, got, expected, history[-15:]))
            if cache.keys() != model.order_mru_to_lru():
                self.fail("order mismatch at op %d (seed=%d cap=%d):\n  %r\n  %r\n%r"
                          % (i, seed, capacity, cache.keys(),
                             model.order_mru_to_lru(), history[-15:]))
            if rng.random() < 0.25:
                clock.advance(rng.choice([0.5, 1.0, 2.0]))
            if i % 500 == 0:
                assert_invariants(self, cache)

        assert_invariants(self, cache)


class TestScale(unittest.TestCase):
    def test_large_workload_stays_correct_and_bounded(self):
        # Checking only the size bound would pass even if get() always missed,
        # so every returned value is checked against the last value written.
        cache = Cache(10_000)
        written = {}
        rng = random.Random(4242)
        hits = 0
        for i in range(200_000):
            key = rng.randrange(30_000)
            if i % 3:
                got = cache.get(key)
                if got != -1:
                    hits += 1
                    self.assertEqual(got, written[key],
                                     "a hit returned a stale value for %r" % key)
            else:
                cache.put(key, i)
                written[key] = i
            self.assertLessEqual(len(cache), 10_000)
        self.assertGreater(hits, 1000, "workload produced too few hits to be meaningful")
        assert_invariants(self, cache)


# ----------------------------------------------------------------------
class TestDemoContract(unittest.TestCase):
    """Guards the screenshot pipeline: the demo must be stable and complete."""

    @staticmethod
    def run_demo(*args):
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [sys.executable, DEMO] + list(args),
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )

    def test_demo_exits_zero_with_no_stderr(self):
        proc = self.run_demo("--section", "all")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")

    def test_demo_output_is_byte_identical_across_runs(self):
        # This test is what licenses the screenshot pipeline: if two runs are
        # identical, a screenshot can never disagree with a real run.
        first = self.run_demo("--section", "all").stdout
        second = self.run_demo("--section", "all").stdout
        self.assertEqual(first, second)

    def test_demo_output_contains_required_evidence(self):
        out = self.run_demo("--section", "all").stdout
        for needle in ('put("A", 10)', 'get("B")', "EVICTED", "[PASS]", "EXPIRED"):
            self.assertIn(needle, out, "demo output is missing %r" % needle)
        self.assertNotIn("[FAIL]", out)

    def test_demo_shows_the_real_miss_value_in_the_returned_column(self):
        # Asserting on the word "EVICTED" proves nothing -- it is a literal in
        # demo.py. These assert on COMPUTED columns, so the tests fail if the
        # cache regresses rather than if the prose changes.
        out = self.run_demo("--section", "basic").stdout
        self.assertRegex(out, r'get\("B"\)\s+-1\s',
                         "the evicted key's get() did not report -1")
        self.assertRegex(out, r'put\("C", 30\)\s+-\s+B\s',
                         "the EVICTED column did not name B")

    def test_demo_ttl_section_shows_a_real_expiry(self):
        out = self.run_demo("--section", "ttl").stdout
        self.assertRegex(out, r'4\.9\s+get\("session:1"\)\s+"alice"',
                         "the entry was not alive just before its deadline")
        self.assertRegex(out, r'5\.0\s+get\("session:1"\)\s+-1\s',
                         "the entry did not actually expire at its deadline")

    def test_every_section_runs_clean_and_fits_eighty_columns(self):
        for args in ([], ["--section", "basic"], ["--section", "eviction"],
                     ["--section", "ttl"], ["--section", "all"],
                     ["--section", "perf"]):
            with self.subTest(args=args):
                proc = self.run_demo(*args)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stderr, "")
                longest = max((len(l) for l in proc.stdout.splitlines()), default=0)
                self.assertLessEqual(longest, 80)

    def test_demo_output_matches_the_committed_screenshot_logs(self):
        # The goldens in photo/ are what the screenshots were rendered from.
        # If the demo drifts from them, the shipped images are stale.
        for section, name in (("basic", "01_assignment_example"),
                              ("eviction", "02_eviction_walkthrough"),
                              ("ttl", "03_ttl_expiration")):
            with self.subTest(section=section):
                golden = os.path.join(ROOT, "photo", name + ".txt")
                if not os.path.exists(golden):
                    self.skipTest("photo/%s.txt not generated yet" % name)
                with open(golden) as fh:
                    expected = fh.read()
                self.assertEqual(
                    self.run_demo("--section", section).stdout, expected,
                    "demo drifted from photo/%s.txt -- re-run `make shots`" % name)

    def test_demo_lines_fit_eighty_columns(self):
        out = self.run_demo("--section", "all").stdout
        longest = max((len(line) for line in out.splitlines()), default=0)
        self.assertLessEqual(longest, 80, "a line would soft-wrap in an 80-column terminal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
