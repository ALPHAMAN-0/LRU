# CLAUDE.md

LRU cache: hash map + doubly linked list, O(1) `get`/`put`, optional TTL.
Python 3.9+, standard library only.

## Build / test / lint

- `python3 demo.py` — runnable demonstration (sections 1-3)
- `python3 -m unittest -v` — 92 tests, no third-party deps
- `make test-all` — run the suite on each Python 3.9+ found; non-zero if any fails
- `make verify` — tests plus `scripts/verify_photos.py` (logs match live runs,
  and .txt/.png hashes match MANIFEST.json)
- `make shots` — regenerate `photo/` (needs Pillow; macOS only)

## Rules observed (in this repo)

- The library, demo and tests import **standard library only**. Pillow is
  confined to `scripts/capture_screenshots.py`. (README.md)
- Code must stay **Python 3.9 compatible** — no PEP 695 generics, no `X | Y`
  at runtime. `make test-all` enforces it. (README.md)
- `demo.py` output must be **byte-identical across runs** and **≤ 80 columns**.
  The screenshot pipeline depends on both; `TestDemoContract` enforces them.
  No wall-clock values, no `id()`, no randomness in demo output. (demo.py)
- Only `_unlink` and `_push_front` may write `prev`/`next` on a real entry;
  `_reset_chain()` is the only place the sentinels are relinked.
  (ARCHITECTURE.md)
- Any change to pointer code must keep `TestDifferentialAgainstReferenceModel`
  green — it compares against an independent `OrderedDict` oracle after every
  operation. (test_lru_cache.py)
- `photo/*.txt` are verbatim program output. Never hand-edit them; regenerate
  with `make shots`. A `.txt` is written only after its `.png` validates, so
  the pair can never come from different runs. (scripts/capture_screenshots.py)
- **No git operations** were performed here by request; the repo is
  deliberately un-initialised. (README.md)

## Read first

- `lru_cache.py` — the implementation; the module docstring lists the invariants
- `ARCHITECTURE.md` — pointer surgery and the six invariants, in detail
- `test_lru_cache.py` — `assert_invariants()` is the contract in executable form
- `TASK3.md` — the Task 3 short answers

Architecture: see ARCHITECTURE.md — read before structural changes
