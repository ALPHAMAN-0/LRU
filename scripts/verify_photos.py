#!/usr/bin/env python3
"""Verify that photo/ is internally consistent and matches live program runs.

Standard library only, so `make verify` stays dependency-free.

Three checks per shot:
  1. the .txt still matches what the program prints right now
     (unittest's own wall-clock duration is normalised away)
  2. the .txt and .png hashes still match MANIFEST.json -- this is what
     catches a stale image left behind by an aborted `make shots`
  3. both files exist and are non-empty

Exits non-zero on any drift.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from hashlib import sha256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO = os.path.join(ROOT, "photo")
DURATION = re.compile(r"in \d+\.\d+s")


def normalise(text):
    return DURATION.sub("in <duration>", text)


def sha256_file(path):
    h = sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest_path = os.path.join(PHOTO, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        print("  photo/MANIFEST.json is missing -- run `make shots`")
        return 1
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    failures = 0
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")

    for entry in manifest:
        name = entry["name"]
        txt = os.path.join(ROOT, entry["txt"])
        png = os.path.join(ROOT, entry["png"])
        problems = []

        for path in (txt, png):
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                problems.append("%s is missing or empty" % os.path.basename(path))
        if problems:
            failures += 1
            print("  %-28s FAIL  %s" % (name, "; ".join(problems)))
            continue

        with open(txt) as fh:
            shipped = fh.read()

        if sha256(shipped.encode()).hexdigest() != entry["stdout_sha256"]:
            problems.append("the .txt no longer matches its manifest hash")
        if "png_sha256" in entry and sha256_file(png) != entry["png_sha256"]:
            problems.append("the .png no longer matches its manifest hash "
                            "(stale image?)")

        args = entry["command"].split()[1:]          # drop the "python3" word
        proc = subprocess.run([sys.executable] + args, cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=300)
        live = proc.stdout + (proc.stderr if "unittest" in entry["command"] else "")
        if normalise(live) != normalise(shipped):
            problems.append("a live run no longer produces this output")

        if problems:
            failures += 1
            print("  %-28s FAIL  %s" % (name, "; ".join(problems)))
        else:
            print("  %-28s OK    log matches a live run; .png matches its hash"
                  % name)

    if failures:
        print("\n%d shot(s) failed. Re-run `make shots`." % failures)
        return 1
    print("\nAll %d screenshots verified against live runs and their hashes."
          % len(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
