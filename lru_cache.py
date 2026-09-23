"""A Least Recently Used (LRU) cache: hash map + doubly linked list.

Both ``get`` and ``put`` run in O(1) amortized time.

Layout
------
The cache keeps two structures that cover each other's weakness:

* ``self._map``  -- a ``dict`` from key to node. Gives O(1) lookup, but a dict
  has no usable notion of "recency".
* a doubly linked list -- gives O(1) reordering and O(1) removal of the tail,
  but has no way to find a node without walking it.

The list is bracketed by two *sentinel* nodes that never hold data and are
never unlinked::

    head <-> MRU <-> ... <-> LRU <-> tail

Sentinels are not decoration. Without them, ``_unlink``/``_push_front`` need
four branches (empty list / first / last / middle); with them there are zero.

Each node stores its own ``key``. That is what makes eviction O(1): once we
have the tail node we can delete its map entry without searching the dict.

Invariants (asserted by ``tests/`` after every operation)
---------------------------------------------------------
1. ``head.prev is None`` and ``tail.next is None`` -- the chain is closed.
2. ``len(self._map)`` == the number of nodes strictly between the sentinels.
3. Walking forward and walking backward visit the same nodes in reverse order.
4. Every node in ``_map.values()`` is reachable from ``head``, and every
   reachable node is in ``_map`` under its own ``key``.
5. ``0 <= len(self._map) <= capacity`` after every public call.
6. ``head.next is tail`` if and only if the cache is empty.

Thread safety
-------------
This class is **not** thread-safe, and the reason is sharper than usual: even
``get`` mutates the linked list, so a reader/writer lock does not help. Safe
concurrent use needs full mutual exclusion on reads as well as writes.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

__all__ = ["LRUCache", "Cache", "CacheStats", "FakeClock", "MISS"]

#: Value returned by :meth:`LRUCache.get` when a key is absent or expired.
#: The assignment specifies -1. See "Known limitations" in README.md for why a
#: caller storing a real -1 should pass an explicit ``default`` instead.
MISS = -1


class _Unset:
    """Type of the private 'argument not supplied' sentinel."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<unset>"


# Distinct from None, because ``ttl=None`` is a meaningful value: it means
# "this entry never expires", overriding any ``default_ttl``.
_UNSET: Any = _Unset()


class CacheStats(NamedTuple):
    """An immutable snapshot of cache counters."""

    hits: int
    misses: int
    evictions: int
    expirations: int
    size: int
    capacity: int

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that hit. 0.0 (not an error) with no lookups."""
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total


class FakeClock:
    """A deterministic stand-in for ``time.monotonic``.

    Injected via ``LRUCache(time_fn=...)`` so TTL behaviour can be tested and
    demonstrated without ``time.sleep``. That makes the tests fast and
    structurally non-flaky rather than merely usually-passing.
    """

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        self._t += float(seconds)
        return self._t

    def set(self, seconds: float) -> float:
        self._t = float(seconds)
        return self._t

    def __repr__(self) -> str:
        return "FakeClock(t=%.3f)" % self._t


class _Node:
    """One cache entry, plus the two sentinels.

    ``__slots__`` removes the per-instance ``__dict__``. At cache scale that
    dict is the dominant memory cost, so this is a real saving, not a flourish.
    """

    __slots__ = ("key", "value", "prev", "next", "expires_at")

    def __init__(
        self,
        key: Any = None,
        value: Any = None,
        expires_at: Optional[float] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None
        self.expires_at = expires_at

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "_Node(key=%r, value=%r, expires_at=%r)" % (
            self.key,
            self.value,
            self.expires_at,
        )


class LRUCache:
    """A fixed-capacity cache that evicts the least recently used entry.

    Args:
        capacity: Maximum number of entries. Must be a positive ``int``.
        default_ttl: Optional lifetime in seconds applied to every ``put``
            that does not pass its own ``ttl``. ``None`` means entries never
            expire.
        time_fn: Zero-argument callable returning a monotonically increasing
            float. Defaults to ``time.monotonic``.

    The module also exports ``Cache`` as an alias of this class, because the
    assignment names the type ``Cache(capacity)``.

    Example:
        >>> cache = LRUCache(2)
        >>> cache.put("A", 10)
        >>> cache.put("B", 20)
        >>> cache.get("A")
        10
        >>> cache.put("C", 30)      # cache was full -> evicts "B"
        >>> cache.get("B")
        -1
    """

    __slots__ = (
        "_capacity",
        "_default_ttl",
        "_time_fn",
        "_map",
        "_head",
        "_tail",
        "_hits",
        "_misses",
        "_evictions",
        "_expirations",
    )

    def __init__(
        self,
        capacity: int,
        *,
        default_ttl: Optional[float] = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        # bool is a subclass of int, so LRUCache(True) would silently mean
        # capacity 1. That is a latent bug, not a convenience: reject it.
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(
                "capacity must be a positive integer, got %s"
                % type(capacity).__name__
            )
        if capacity <= 0:
            raise ValueError(
                "capacity must be a positive integer, got %d" % capacity
            )
        if not callable(time_fn):
            raise TypeError(
                "time_fn must be a zero-argument callable returning a float, "
                "got %s" % type(time_fn).__name__
            )

        self._capacity = capacity
        self._default_ttl = (
            None if default_ttl is None else self._validate_ttl(default_ttl)
        )
        self._time_fn = time_fn

        self._map: Dict[Any, _Node] = {}
        self._head = _Node()  # sentinel: head.next is the MRU entry
        self._tail = _Node()  # sentinel: tail.prev is the LRU entry
        self._reset_chain()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of entries. Fixed at construction."""
        return self._capacity

    @property
    def default_ttl(self) -> Optional[float]:
        """Lifetime in seconds applied to puts with no explicit ``ttl``."""
        return self._default_ttl

    @property
    def time_fn(self) -> Callable[[], float]:
        """The injected clock. ``time.monotonic`` unless overridden."""
        return self._time_fn

    @property
    def lru_key(self) -> Any:
        """Key of the least recently used resident entry, or ``None`` if empty.

        This is the entry an insert would remove next. Exposed so callers can
        name the victim without reaching into internals. O(1).
        """
        node = self._tail.prev
        return None if node is self._head else node.key

    @property
    def stats(self) -> CacheStats:
        """Snapshot of the lifetime counters. O(1)."""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            expirations=self._expirations,
            size=len(self._map),
            capacity=self._capacity,
        )

    # ------------------------------------------------------------------
    # Linked-list surgery. _unlink and _push_front are the only places that
    # touch the pointers of a REAL entry; _reset_chain relinks the sentinels
    # and is used only by __init__ and clear().
    # ------------------------------------------------------------------

    def _reset_chain(self) -> None:
        """Point the two sentinels at each other. The empty-list state."""
        self._head.prev = None
        self._head.next = self._tail
        self._tail.prev = self._head
        self._tail.next = None

    def _unlink(self, node: _Node) -> None:
        """Detach ``node`` from the chain. O(1), branch-free thanks to sentinels."""
        node.prev.next = node.next
        node.next.prev = node.prev
        # Defensive: a detached node cannot corrupt the chain if it is
        # accidentally reused.
        node.prev = None
        node.next = None

    def _push_front(self, node: _Node) -> None:
        """Insert ``node`` directly after the head sentinel (as MRU). O(1)."""
        first = self._head.next
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node

    def _move_to_front(self, node: _Node) -> None:
        """Promote ``node`` to most-recently-used. O(1)."""
        if self._head.next is node:
            return  # already MRU; skip six pointer writes
        self._unlink(node)
        self._push_front(node)

    def _remove_node(self, node: _Node) -> None:
        """Unlink ``node`` and drop its map entry. O(1)."""
        self._unlink(node)
        del self._map[node.key]

    def _evict_lru(self) -> _Node:
        """Evict and return the least recently used entry. O(1)."""
        victim = self._tail.prev
        if victim is self._head:  # pragma: no cover - guarded by callers
            raise RuntimeError("internal error: eviction from an empty cache")
        self._remove_node(victim)
        self._evictions += 1
        return victim

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ttl(ttl: Any) -> float:
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)):
            raise TypeError(
                "ttl must be a positive number of seconds, got %s"
                % type(ttl).__name__
            )
        if ttl <= 0:
            raise ValueError(
                "ttl must be a positive number of seconds, got %r" % (ttl,)
            )
        return float(ttl)

    def _resolve_expiry(self, ttl: Any) -> Optional[float]:
        """Turn a ``ttl`` argument into an absolute deadline (or ``None``).

        Three-state argument:
          ``_UNSET`` -> inherit ``default_ttl``
          ``None``   -> never expires, overriding ``default_ttl``
          a number   -> that many seconds from now
        """
        if ttl is _UNSET:
            ttl = self._default_ttl
        if ttl is None:
            return None
        return self._time_fn() + self._validate_ttl(ttl)

    def _is_expired(self, node: _Node, now: Optional[float] = None) -> bool:
        """True once the clock reaches the deadline.

        Uses ``>=``: a ttl of ``t`` defines the half-open validity window
        ``[written, written + t)``, so the entry is dead at exactly the
        deadline. Arbitrary, but chosen, documented, and pinned by a test.

        ``now`` lets an O(n) walk sample the clock once and classify every
        node against the same instant, instead of re-reading a moving clock
        per node and producing a result that describes n different moments.
        """
        if node.expires_at is None:
            return False
        return (self._time_fn() if now is None else now) >= node.expires_at

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: Any, default: Any = MISS) -> Any:
        """Return the value for ``key``, else ``default`` (-1 by default).

        A successful get makes ``key`` the most recently used entry. A miss
        changes nothing about the ordering. O(1) amortized.
        """
        node = self._map.get(key)
        if node is None:
            self._misses += 1
            return default
        if self._is_expired(node):
            # Lazy reclamation: the expired entry is dropped on the first
            # access that observes it, which keeps this path O(1).
            self._remove_node(node)
            self._expirations += 1
            self._misses += 1
            return default
        self._hits += 1
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any, *, ttl: Any = _UNSET) -> None:
        """Insert or update ``key``, evicting the LRU entry only if needed.

        Updating a key that is already present never evicts anything: the
        entry already owns a capacity slot. O(1) amortized.
        """
        expires_at = self._resolve_expiry(ttl)
        node = self._map.get(key)

        if node is not None:
            # UPDATE path. Reuses the node, so it cannot overflow capacity.
            # An expired-but-still-resident key lands here too: it is a
            # refresh, not an insert, and is never counted as an eviction.
            node.value = value
            node.expires_at = expires_at
            self._move_to_front(node)
            return

        # INSERT path -- the only branch that can evict.
        if len(self._map) >= self._capacity:
            # Prefer reclaiming already-dead entries over discarding live
            # ones: recency and liveness are independent, so the tail may be
            # a corpse while a live entry would otherwise be thrown away.
            # Each entry is reclaimed at most once, so this stays O(1)
            # amortised, and it attributes the removal to TTL rather than to
            # capacity pressure.
            now = self._time_fn()
            while (len(self._map) >= self._capacity
                   and self._is_expired(self._tail.prev, now)):
                self._remove_node(self._tail.prev)
                self._expirations += 1
            if len(self._map) >= self._capacity:
                self._evict_lru()
        node = _Node(key, value, expires_at)
        self._map[key] = node
        self._push_front(node)

    def peek(self, key: Any, default: Any = MISS) -> Any:
        """Return the value for ``key`` without touching recency.

        Pure: no reordering, no reclamation, no counter changes. O(1).
        """
        node = self._map.get(key)
        if node is None or self._is_expired(node):
            return default
        return node.value

    def keys(self) -> List[Any]:
        """Live keys ordered most-recently-used first. O(n)."""
        return [key for key, _ in self.items()]

    def items(self) -> List[Tuple[Any, Any]]:
        """Live ``(key, value)`` pairs ordered most-recently-used first. O(n)."""
        out: List[Tuple[Any, Any]] = []
        now = self._time_fn()
        node = self._head.next
        while node is not self._tail:
            if not self._is_expired(node, now):
                out.append((node.key, node.value))
            node = node.next
        return out

    def purge_expired(self) -> int:
        """Drop every expired entry and return how many were removed. O(n).

        The only non-O(1) maintenance operation. Call it periodically if the
        workload writes short-TTL keys that are never read again -- those
        entries hold their capacity slot until something observes them.
        """
        removed = 0
        now = self._time_fn()
        node = self._head.next
        while node is not self._tail:
            nxt = node.next
            if self._is_expired(node, now):
                self._remove_node(node)
                self._expirations += 1
                removed += 1
            node = nxt
        return removed

    def clear(self) -> None:
        """Drop every entry. Lifetime counters are preserved. O(n).

        The discarded nodes are unlinked as we go. Skipping that would leave
        them pointing at each other -- a reference cycle that keeps every
        cached value alive until the cyclic collector happens to run, rather
        than freeing them immediately on refcount.
        """
        node = self._head.next
        while node is not self._tail:
            nxt = node.next
            node.prev = None
            node.next = None
            node = nxt
        self._map.clear()
        self._reset_chain()

    # ------------------------------------------------------------------
    # Dunder protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of *resident* entries, including any not-yet-observed
        expired ones.

        ``len`` answers a capacity question ("what is occupying space right
        now"); ``in`` answers a liveness question ("what would a lookup
        find"). Keeping ``len`` O(1) is worth that asymmetry, and it is
        documented rather than hidden. Use ``purge_expired()`` first for an
        exact live count.
        """
        return len(self._map)

    def __contains__(self, key: Any) -> bool:
        """True if ``key`` is present and not expired. Pure; O(1)."""
        node = self._map.get(key)
        return node is not None and not self._is_expired(node)

    def __iter__(self) -> Iterator[Any]:
        """Iterate live keys most-recently-used first."""
        return iter(self.keys())

    def __repr__(self) -> str:
        keys = self.keys()
        preview = ", ".join(repr(k) for k in keys[:8])
        if len(keys) > 8:
            preview += ", ... (+%d more)" % (len(keys) - 8)
        return "%s(capacity=%d, size=%d, mru_to_lru=[%s])" % (
            type(self).__name__,
            self._capacity,
            len(self._map),
            preview,
        )


#: The assignment names the class ``Cache``; this alias keeps the demo output
#: mapping one-to-one onto the spec text without duplicating the implementation.
Cache = LRUCache
