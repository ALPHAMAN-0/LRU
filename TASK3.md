# Task 3 — Algorithm Explanation & Critical Thinking

**1. Data structures, and why.**
A hash map from key to node, plus a doubly linked list with sentinel head and
tail. The map finds entries in O(1) but ignores recency; the list reorders and
drops its tail in O(1) but cannot search. Nodes store their own key, so
eviction needs no lookup.

**2. Time and space complexity.**
`get` and `put` are O(1) amortised: one hash lookup plus a fixed number of
pointer writes. Amortised, not worst case, because dict resizing rehashes and a
lazy TTL unlink can fire once per entry. Space is O(capacity) — one map entry
and one node per item.

**3. Where it performs poorly.**
A sequential scan longer than the cache: a nightly job reading keys 1..N, N
above capacity, evicts each key just before it is needed again, so the hit rate
is 0%. LRU has no scan resistance; ARC does.

**4. An AI suggestion I rejected.**
It reached for `collections.OrderedDict.move_to_end` — correct, but it hides the
pointer mechanics this task exists to demonstrate. I hand-wrote the list,
keeping `OrderedDict` as the oracle in a randomised differential test.
