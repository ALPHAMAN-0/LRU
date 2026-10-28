#!/usr/bin/env python3
"""Runnable demonstration of lru_cache.LRUCache.

Every line of output is produced by executing real cache operations -- the
tables are rendered from ``cache.keys()`` after each call, never from a
hand-maintained shadow copy. The output is byte-for-byte identical on every
run (no wall clock, no addresses, no randomness), which is what lets the
screenshots in photo/ be verified with a plain ``diff``.

Usage:
    python3 demo.py                       # sections 1-3
    python3 demo.py --section basic       # the assignment example only
    python3 demo.py --section eviction
    python3 demo.py --section ttl
    python3 demo.py --section perf        # timings; not part of "all"
"""

from __future__ import annotations

import utils.net
import utils.clients
import utils.http_session
import utils.transport
import utils.session
import argparse
import sys
import time

from lru_cache import Cache, FakeClock

WIDTH = 80


# ----------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------

def q(value: object) -> str:
    """Render a value the way the assignment brief writes it (double quotes)."""
    if isinstance(value, str):
        return '"%s"' % value
    return repr(value)


def banner(title: str, subtitle: str = "") -> None:
    print("=" * WIDTH)
    line = "  " + title
    if subtitle:
        pad = WIDTH - len(line) - len(subtitle) - 2
        line = line + " " * max(1, pad) + subtitle
    print(line)
    print("=" * WIDTH)
    print()


def order_of(cache) -> str:
    pairs = ["%s=%s" % (k, v) for k, v in cache.items()]
    return ", ".join(pairs) if pairs else "(empty)"


# ----------------------------------------------------------------------
# section 1 + 2: a step table driven by real operations
# ----------------------------------------------------------------------

HEAD = "  STEP  OPERATION            RETURNED  EVICTED  CACHE AFTER (MRU -> LRU)"
RULE = "  ----  -------------------  --------  -------  ---------------------------"
ROW = "  {step:>4}  {op:<19}  {ret:<8}  {ev:<7}  {order}"


class StepTable:
    """Executes cache operations and prints one row per operation."""

    def __init__(self, cache) -> None:
        self.cache = cache
        self.step = 0
        self.returns = []

    def start(self) -> None:
        print(HEAD)
        print(RULE)

    def finish(self) -> None:
        print(RULE)
        print()

    def _row(self, op: str, ret: str, ev: str, note: str = "") -> None:
        self.step += 1
        print(ROW.format(step=self.step, op=op, ret=ret, ev=ev,
                         order=order_of(self.cache)))
        if note:
            for line in note.split("\n"):
                print("        " + line)

    def put(self, key, value, note: str = "") -> None:
        # Ask the cache which entry is next in line rather than inferring it
        # from a set difference: keys() hides expired-but-resident entries,
        # so a diff would name the wrong key once TTLs are in play.
        victim = self.cache.lru_key
        size_before = len(self.cache)
        evictions_before = self.cache.stats.evictions
        self.cache.put(key, value)
        evicted = "-"
        if self.cache.stats.evictions > evictions_before:
            evicted = str(victim) if victim is not None else "?"
            note = ('>>> EVICTED %s -- it was the least-recently-used entry,\n'
                    '    and the cache was full (size %d of capacity %d).'
                    % (q(evicted), size_before, self.cache.capacity)
                    ) + ("\n" + note if note else "")
        self._row("put(%s, %s)" % (q(key), q(value)), "-", evicted, note)

    def get(self, key, note: str = "") -> None:
        result = self.cache.get(key)
        self.returns.append(result)
        self._row("get(%s)" % q(key), q(result), "-", note)

    def print_stats(self) -> None:
        s = self.cache.stats
        print("  stats: hits=%d misses=%d evictions=%d expirations=%d "
              "size=%d capacity=%d" % (s.hits, s.misses, s.evictions,
                                       s.expirations, s.size, s.capacity))
        print()


def section_basic() -> None:
    banner("LRU CACHE  -  SECTION 1: ASSIGNMENT EXAMPLE", "capacity = 2")
    print("  cache = Cache(2)")
    print()

    table = StepTable(Cache(2))
    table.start()
    table.put("A", 10)
    table.put("B", 20)
    table.get("A", '>>> HIT: "A" is returned AND promoted to most-recently-used,\n'
                   '    so "B" is now the eviction candidate.')
    table.put("C", 30)
    table.get("B", '>>> MISS: "B" was evicted, so get() returns the sentinel -1.')
    table.get("C")
    table.get("A")
    table.finish()

    expected = [10, -1, 30, 10]
    verdict = "[PASS]" if table.returns == expected else "[FAIL]"
    print("  SPEC CHECK  expected returns: " + ", ".join(str(v) for v in expected))
    print("              actual   returns: " + ", ".join(str(v) for v in table.returns)
          + "        " + verdict)
    print()
    table.print_stats()
    if table.returns != expected:
        raise SystemExit("demo failed its own spec check")


def section_eviction() -> None:
    banner("LRU CACHE  -  SECTION 2: EVICTION WALKTHROUGH", "capacity = 3")
    print("  cache = Cache(3)")
    print()

    table = StepTable(Cache(3))
    table.start()
    table.put("A", 10)
    table.put("B", 20)
    table.put("C", 30)
    table.get("A", '>>> "A" jumps to the front; "B" is now least-recently-used.')
    table.put("D", 40)
    table.put("C", 99, '>>> NO EVICTION: "C" already existed, so put() updated it\n'
                       '    in place. An update never costs a capacity slot.')
    table.get("Z", '>>> A MISS IS INERT: nothing was promoted, so the order above\n'
                   '    is unchanged and the next victim is still "A".')
    table.put("E", 50)
    table.finish()
    table.print_stats()


# ----------------------------------------------------------------------
# section 3: TTL / expiration, on an injected deterministic clock
# ----------------------------------------------------------------------

TTL_HEAD = "  TIME  OPERATION                     RETURNED  NOTE"
TTL_RULE = "  ----  ----------------------------  --------  ----------------------------"
TTL_ROW = "  {t:>4}  {op:<28}  {ret:<8}  {note}"


class TtlTable:
    def __init__(self, cache, clock) -> None:
        self.cache = cache
        self.clock = clock

    def start(self) -> None:
        print(TTL_HEAD)
        print(TTL_RULE)

    def finish(self) -> None:
        print(TTL_RULE)
        print()

    def row(self, op: str, ret: str, note: str) -> None:
        print(TTL_ROW.format(t="%.1f" % self.clock(), op=op, ret=ret, note=note))

    def put(self, key, value, ttl="inherit", note: str = "") -> None:
        if ttl == "inherit":
            self.cache.put(key, value)
            deadline = self.clock() + (self.cache.default_ttl or 0)
            note = note or ("expires at t=%.1f" % deadline
                            if self.cache.default_ttl else "no expiry")
            op = "put(%s, %s)" % (q(key), q(value))
        else:
            self.cache.put(key, value, ttl=ttl)
            op = "put(%s, %s, ttl=%s)" % (q(key), q(value), ttl)
            note = note or ("expires at t=%.1f" % (self.clock() + ttl)
                            if ttl is not None else "never expires")
        self.row(op, "-", note)

    def get(self, key, note: str) -> object:
        result = self.cache.get(key)
        self.row("get(%s)" % q(key), q(result), note)
        return result


def section_ttl() -> None:
    banner("LRU CACHE  -  SECTION 3: TTL / EXPIRATION", "optional feature")
    print("  clock = FakeClock(0.0)        # injected, so this run is repeatable")
    print("  cache = Cache(3, default_ttl=5.0, time_fn=clock)")
    print()

    clock = FakeClock(0.0)
    cache = Cache(3, default_ttl=5.0, time_fn=clock)
    table = TtlTable(cache, clock)

    table.start()
    table.put("session:1", "alice")
    table.get("session:1", "age 0.0s of 5.0s TTL")
    clock.set(4.9)
    table.get("session:1", "age 4.9s of 5.0s TTL")
    clock.set(5.0)
    table.get("session:1", "EXPIRED (5.0 >= 5.0 deadline)")
    table.finish()

    s = cache.stats
    print("  >>> The entry was reclaimed lazily on that read: size is now %d and"
          % s.size)
    print("      stats.expirations is now %d. get() still ran in O(1)." % s.expirations)
    print()

    print("  " + "-" * (WIDTH - 4))
    print("  Per-key TTL, and an entry that opts out of the default entirely")
    print("  " + "-" * (WIDTH - 4))
    clock.set(0.0)
    cache = Cache(3, default_ttl=5.0, time_fn=clock)
    table = TtlTable(cache, clock)
    table.start()
    table.put("short", 1)
    table.put("long", 2, ttl=100.0)
    table.put("forever", 3, ttl=None, note="never expires (opts out)")
    clock.set(6.0)
    table.get("short", "EXPIRED at t=5.0")
    table.get("long", "alive until t=100.0")
    table.get("forever", "no deadline was ever set")
    table.finish()

    print("  " + "-" * (WIDTH - 4))
    print("  len() vs `in`: a capacity question and a liveness question")
    print("  " + "-" * (WIDTH - 4))
    clock.set(0.0)
    cache = Cache(3, default_ttl=2.0, time_fn=clock)
    cache.put("ghost", "value")
    clock.set(3.0)
    print("    After the TTL passes, but before anything reads the key:")
    print("      len(cache)        -> %d   (it still occupies a capacity slot)"
          % len(cache))
    print("      'ghost' in cache  -> %s   (a lookup would not find it)"
          % ("ghost" in cache))
    print("      cache.keys()      -> %r" % (cache.keys(),))
    print("    purge_expired() sweeps them out on demand:")
    print("      cache.purge_expired() -> %d" % cache.purge_expired())
    print("      len(cache)            -> %d" % len(cache))
    print()
    print("  This asymmetry is deliberate: keeping len() O(1) is worth more than")
    print("  making it liveness-exact, which would cost an O(n) scan per call.")
    print()


# ----------------------------------------------------------------------
# section 4: empirical support for the O(1) claim (excluded from "all")
# ----------------------------------------------------------------------

def section_perf() -> None:
    banner("LRU CACHE  -  SECTION 4: EMPIRICAL O(1) CHECK", "timings vary")
    print("  Per-operation cost should stay flat as the cache grows 100x.")
    print()
    print("  CAPACITY   OPERATIONS   TOTAL (ms)   PER OP (ns)")
    print("  ---------  -----------  -----------  -----------")
    for capacity in (1_000, 10_000, 100_000):
        cache = Cache(capacity)
        ops = 200_000
        start = time.perf_counter()
        for i in range(ops):
            # Read a DIFFERENT key from the one just written, or every get
            # would hit _move_to_front's already-MRU early return and the
            # benchmark would never time the relink path it claims to.
            cache.put(i % (capacity * 2), i)
            cache.get((i + capacity // 2) % (capacity * 2))
        elapsed = time.perf_counter() - start
        print("  %9d  %11d  %11.1f  %11.0f"
              % (capacity, ops * 2, elapsed * 1000, elapsed / (ops * 2) * 1e9))
    print("  ---------  -----------  -----------  -----------")
    print()
    print("  Flat per-op cost across a 100x capacity range is what O(1) looks")
    print("  like in practice. Absolute numbers are machine-dependent.")
    print()


SECTIONS = {
    "basic": section_basic,
    "eviction": section_eviction,
    "ttl": section_ttl,
    "perf": section_perf,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate the LRU cache implementation.")
    parser.add_argument(
        "--section", default="all",
        choices=["all", "basic", "eviction", "ttl", "perf"],
        help="which section to run (default: all; 'perf' is excluded from 'all' "
             "because timings are not reproducible)")
    args = parser.parse_args(argv)

    if args.section == "all":
        section_basic()
        section_eviction()
        section_ttl()
    else:
        SECTIONS[args.section]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
