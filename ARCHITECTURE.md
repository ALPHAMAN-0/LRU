# Architecture

Read this before changing anything in `lru_cache.py`. The cache is small, but
every method depends on six invariants that are easy to break and silent when
broken — which is why the test suite re-asserts all of them after mixed
operation sequences.

## The shape

```
                  _map: dict[key] -> _Node
                    |      |      |
                    v      v      v
   head <----> [ C=30 ] <-> [ A=10 ] <-> [ B=20 ] <----> tail
  sentinel        MRU                       LRU        sentinel
```

- `_map` answers "where is key K?" in O(1).
- The list answers "who was used longest ago?" in O(1) — it is always
  `tail.prev`.
- `head` and `tail` are `_Node` instances that hold no data, live outside
  `_map`, and are never unlinked.

## Invariants

1. `head.prev is None` and `tail.next is None` — the chain is closed at both
   ends.
2. `len(_map)` equals the number of nodes strictly between the sentinels.
3. Walking forward from `head` and backward from `tail` visits the same nodes
   in reverse order.
4. Every node in `_map.values()` is reachable from `head`, and every reachable
   node is in `_map` under its own `key`.
5. `0 <= len(_map) <= capacity` after every public call. `put` evicts
   *before* it inserts, so the upper bound is never exceeded even
   transiently.
6. `head.next is tail` if and only if the cache is empty.

`assert_invariants()` in `test_lru_cache.py` checks all six. Any change to the
pointer code should be run against `TestDifferentialAgainstReferenceModel`,
which compares against an independent `OrderedDict` model after every
operation.

## Pointer surgery

`_unlink` and `_push_front` are the only methods that write the pointers of a
real entry; everything else composes them. A third helper, `_reset_chain()`,
relinks the two sentinels into the empty-list state and is used only by
`__init__` and `clear()`. Those are the entire risk surface.

**Unlink** — detach a node. Both neighbours are guaranteed to exist, because
the sentinels mean a real node can never be first or last:

```python
node.prev.next = node.next
node.next.prev = node.prev
node.prev = node.next = None      # defensive: a stale node can't corrupt the chain
```

**Push front** — insert directly after `head`, making it MRU:

```python
first = self._head.next
node.prev, node.next = self._head, first
self._head.next = node
first.prev = node
```

**Move to front** = unlink + push front, with an early return when the node is
already MRU (skipping six pointer writes on the hot path).

**Evict** = take `tail.prev`, unlink it, `del _map[node.key]`. The node carries
its own key, which is the only reason this is O(1) rather than a dict search.

## Operation paths

| Path | Sequence |
|---|---|
| `get` hit | `_map.get` → expiry check → `_move_to_front` → return value |
| `get` miss | `_map.get` → count a miss → return the default. **No pointer writes.** |
| `get` expired | `_map.get` → expiry check → `_remove_node` → return the default |
| `put` update | `_map.get` → set value/deadline → `_move_to_front`. **Never evicts.** |
| `put` insert | `_map.get` → evict `tail.prev` if full → new node → `_push_front` |

The eviction check lives *only* on the insert branch. That is the whole
mechanism behind "updating an existing key must not evict anything", and it is
the single most commonly failed edge case in this problem.

## Where TTL fits

`expires_at` is one extra slot on `_Node`: `None`, or an absolute deadline on
the injected clock. It never changes the list structure. Expiry is observed
lazily — the first `get` to touch a dead entry unlinks it, and an insert
needing room reclaims dead entries from the LRU end before evicting a live one
— so no operation loses its amortised O(1) bound. `purge_expired()` is the only
O(n) *maintenance* operation; the read-out helpers `keys()`, `items()`,
`__iter__` and `__repr__` also walk the chain, as does `clear()`.

See README.md for the full behaviour matrix of an expired-but-resident entry.
