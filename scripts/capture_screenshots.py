#!/usr/bin/env python3
"""Generate the screenshots in photo/ from REAL program runs.

Design goal: it must be impossible for a screenshot to show text that the
program did not actually print.

    1. Each shot runs the program ONCE as a subprocess and keeps the verbatim
       stdout in memory.
    2. That exact text is written to photo/<name>.txt, unmodified and with no
       added header, so anyone can verify the image:

           python3 demo.py --section basic | diff - photo/01_assignment_example.txt

    3. The same in-memory text is what gets photographed. Three paths are
       tried in order, and every one of them is validated on the pixels
       before it is accepted:

       A. a genuine screen capture of a real Terminal.app window
       B. headless Google Chrome rendering the verbatim text
       C. Pillow drawing the verbatim text directly

Why the pixel validation matters: on macOS, if Screen Recording permission is
missing, `screencapture` still exits 0 and hands back the desktop wallpaper
with no windows in it. The exit code is useless as a signal. So every capture
path paints the same sentinel background (#123456 -- a colour no wallpaper is
mostly made of) and the validator requires it to dominate the image.

Usage:
    python3 scripts/capture_screenshots.py
    python3 scripts/capture_screenshots.py --html-only   # skip Terminal capture
    python3 scripts/capture_screenshots.py --keep-window # leave Terminal open

Requires Pillow (pip install Pillow). The library, demo and tests do not.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from hashlib import sha256

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: python3 -m pip install Pillow")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO = os.path.join(ROOT, "photo")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MENLO = "/System/Library/Fonts/Menlo.ttc"

SENTINEL_BG = (0x12, 0x34, 0x56)
FOREGROUND = (0xE6, 0xED, 0xF3)

# Terminal.app wants 16-bit channels.
def _t16(rgb):
    return "{%d, %d, %d}" % tuple(round(c / 255 * 65535) for c in rgb)


class Shot:
    def __init__(self, name, args, title, caption, merge_stderr=False):
        self.name = name
        self.args = args            # passed to sys.executable
        self.title = title          # shown in the window title bar
        self.caption = caption
        self.merge_stderr = merge_stderr


SHOTS = [
    Shot("01_assignment_example", ["demo.py", "--section", "basic"],
         "python3 demo.py --section basic",
         "put(), get(), LRU eviction and every returned value"),
    Shot("02_eviction_walkthrough", ["demo.py", "--section", "eviction"],
         "python3 demo.py --section eviction",
         "recency promotion, update-without-eviction, and an inert miss"),
    Shot("03_ttl_expiration", ["demo.py", "--section", "ttl"],
         "python3 demo.py --section ttl",
         "TTL expiration on an injected deterministic clock"),
    Shot("04_unit_tests", ["-m", "unittest"],
         "python3 -m unittest",
         "the full test suite passing", merge_stderr=True),
]


class CaptureError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# 1. run the program and keep the verbatim output
# ----------------------------------------------------------------------

def run_and_capture(shot):
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable] + shot.args,
        cwd=ROOT, env=env, text=True, timeout=300,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if shot.merge_stderr else subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise CaptureError(
            "%s exited %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
            % (shot.name, proc.returncode, proc.stdout, proc.stderr or ""))
    if not proc.stdout.strip():
        raise CaptureError("%s produced no output to photograph" % shot.name)
    if not shot.merge_stderr and proc.stderr:
        raise CaptureError("%s wrote to stderr:\n%s" % (shot.name, proc.stderr))
    return proc.stdout


# ----------------------------------------------------------------------
# 2. pixel validation -- the only reliable failure detector on macOS
# ----------------------------------------------------------------------

def _bg_fraction(im, colour, tol):
    r, g, b = im.split()
    total = im.size[0] * im.size[1]
    mask = None
    for chan, target in zip((r, g, b), colour):
        lo, hi = max(0, target - tol), min(255, target + tol)
        band = chan.point(lambda v, lo=lo, hi=hi: 255 if lo <= v <= hi else 0)
        mask = band if mask is None else ImageChops.multiply(mask, band)
    # histogram()[255] instead of sum(getdata()): forward-compatible with
    # Pillow 14 (getdata is deprecated) and far faster.
    return mask.histogram()[255] / float(total)


def validate_png(path, tol=26, min_bg=0.25, min_ink=0.01, max_ink=0.75,
                 expected_lines=None, bar_px=0):
    im = Image.open(path).convert("RGB")
    im.load()
    w, h = im.size
    bg = _bg_fraction(im, SENTINEL_BG, tol)
    stddev = ImageStat.Stat(im.convert("L")).stddev[0]
    colours = im.getcolors(maxcolors=2_000_000)
    distinct = len(colours) if colours else 2_000_000
    report = {
        "size": [w, h],
        "bg_match": round(bg, 4),
        "stddev": round(stddev, 2),
        "distinct_colors": distinct,
    }
    # Measure ink only BELOW the window chrome. The title bar and its dots are
    # non-sentinel pixels by construction, so a whole-image ink ratio would
    # happily "validate" an image containing no program output at all.
    body = im.crop((0, min(bar_px, h - 1), w, h)) if bar_px else im
    body_ink = 1.0 - _bg_fraction(body, SENTINEL_BG, tol)
    text_rows = _ink_row_count(body, SENTINEL_BG, tol)
    report["body_ink"] = round(body_ink, 4)
    report["text_rows"] = text_rows
    checks = {
        "dimensions": w >= 400 and h >= 200,
        "not_flat": stddev >= 6.0 and distinct >= 40,
        "sentinel_background": bg >= min_bg,
        "text_present": min_ink <= body_ink <= max_ink,
        "not_clipped_right": not _touches_right_edge(
            im, SENTINEL_BG, tol, top=bar_px + 24, bottom=24),
    }
    if expected_lines is not None:
        # A truncated render is the one failure a pretty image can hide, so
        # require roughly one band of ink per non-blank line of output.
        non_blank = max(1, expected_lines)
        checks["line_count_plausible"] = text_rows >= int(non_blank * 0.5)
        report["expected_lines"] = expected_lines
    failed = [name for name, ok in checks.items() if not ok]
    report["failed_checks"] = failed
    report["ok"] = not failed
    return report


def _ink_row_count(im, colour, tol):
    """Number of horizontal bands that contain non-background pixels."""
    w, h = im.size
    bands, run = 0, False
    for y in range(h):
        row = im.crop((0, y, w, y + 1))
        inked = (1.0 - _bg_fraction(row, colour, tol)) > 0.002
        if inked and not run:
            bands += 1
        run = inked
    return bands


def _touches_right_edge(im, colour, tol, margin=3, top=0, bottom=0):
    """True if text runs into the right edge, i.e. the render was clipped.

    The frame has rounded corners, so the extreme corners of the edge strip
    are page background rather than terminal background. Sampling only the
    text region between ``top`` and ``h - bottom`` avoids that false positive.
    """
    w, h = im.size
    lo, hi = min(top, h - 1), max(0, h - bottom)
    if w <= margin or hi <= lo:
        return False
    strip = im.crop((w - margin, lo, w, hi))
    return (1.0 - _bg_fraction(strip, colour, tol)) > 0.002


PERMISSION_HELP = """\
    The window content was not present in the captured pixels. On macOS this
    almost always means Screen Recording permission is missing for whichever
    app is running this script (screencapture exits 0 even when denied).
    Grant it under:
      System Settings > Privacy & Security > Screen & System Audio Recording
    then quit and relaunch that app. Falling back to a headless-Chrome render
    of the same verbatim output."""


# ----------------------------------------------------------------------
# 3. Path A -- a real Terminal.app window
# ----------------------------------------------------------------------

def screen_recording_available():
    """Probe whether screencapture can actually read the screen.

    Cheap and side-effect free: grab an 8x8 rect. When Screen Recording
    permission is missing, macOS fails this with "could not create image
    from rect", which saves us from opening and closing a Terminal window
    only to discover the same thing 30 seconds later.
    """
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    os.unlink(tmp)
    try:
        proc = subprocess.run(["screencapture", "-x", "-R0,0,8,8", tmp],
                              capture_output=True, text=True, timeout=30)
        blocked = ("could not create image" in (proc.stderr or "").lower())
        if proc.returncode != 0 or blocked or not os.path.exists(tmp):
            return False, ((proc.stderr or "").strip()
                           or "screencapture returned no image")
        return True, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _osascript(script):
    proc = subprocess.run(["osascript", "-e", script],
                          capture_output=True, text=True, timeout=90)
    return proc


def _applescript_denied(proc):
    return "-1743" in proc.stderr or "Not authorized" in proc.stderr


def try_terminal_capture(shot, raw, png_path, keep_window=False):
    lines = raw.splitlines()
    cols = max(84, min(200, max((len(l) for l in lines), default=80) + 4))
    rows = min(70, len(lines) + 4)
    if len(lines) + 4 > 70:
        return None, "output is taller than a capturable window"

    workdir = tempfile.mkdtemp(prefix="lru-tee-")
    tee = os.path.join(workdir, "live.txt")
    redirect = " 2>&1" if shot.merge_stderr else ""
    cmd = "cd %s && clear && %s %s%s | tee %s" % (
        _sh(ROOT), _sh(sys.executable),
        " ".join(_sh(a) for a in shot.args), redirect, _sh(tee))

    script = """
    tell application "Terminal"
        activate
        set newTab to do script ""
        delay 0.4
        set w to window 1
        set winID to id of w
        set background color of newTab to %s
        set normal text color of newTab to %s
        set bold text color of newTab to %s
        set cursor color of newTab to %s
        set font name of newTab to "Menlo"
        set font size of newTab to 14
        set number of columns of newTab to %d
        set number of rows of newTab to %d
        set position of w to {60, 60}
        delay 0.3
        do script %s in newTab
        return winID as string
    end tell
    """ % (_t16(SENTINEL_BG), _t16(FOREGROUND), _t16(FOREGROUND),
           _t16(FOREGROUND), cols, rows, _as_str(cmd))

    proc = _osascript(script)
    if proc.returncode != 0:
        if _applescript_denied(proc):
            return None, ("Automation permission denied for Terminal "
                          "(System Settings > Privacy & Security > Automation)")
        return None, "osascript failed: %s" % proc.stderr.strip()

    win_id = proc.stdout.strip()
    try:
        geometry = _osascript(
            'tell application "Terminal" to get {number of columns, '
            'number of rows} of selected tab of window id %s' % win_id)
        if geometry.returncode == 0:
            try:
                got_cols, got_rows = [int(v) for v in
                                      geometry.stdout.strip().split(", ")]
                if got_cols < cols or got_rows < rows:
                    return None, ("Terminal clamped the window to %dx%d "
                                  "(needed %dx%d) -- output would scroll off "
                                  "the top" % (got_cols, got_rows, cols, rows))
            except ValueError:
                pass
        if not _wait_until_idle(win_id, timeout=120):
            return None, "the command in Terminal did not finish in time"
        time.sleep(0.6)  # let the window finish drawing

        if not os.path.exists(tee):
            return None, "the Terminal run produced no output file"
        live = open(tee, "r").read()
        if _normalise(live) != _normalise(raw):
            return None, ("the Terminal run did not match the captured run "
                          "(refusing to ship a divergent image)")

        for attempt in ("window", "region"):
            if attempt == "window":
                cap = subprocess.run(
                    ["screencapture", "-x", "-o", "-l" + win_id, png_path],
                    capture_output=True, text=True, timeout=60)
            else:
                bounds = _osascript(
                    'tell application "Terminal" to get bounds of window id %s'
                    % win_id)
                if bounds.returncode != 0:
                    continue
                nums = [int(float(v)) for v in bounds.stdout.strip().split(", ")]
                left, top, right, bottom = nums
                cap = subprocess.run(
                    ["screencapture", "-x",
                     "-R%d,%d,%d,%d" % (left, top, right - left, bottom - top),
                     png_path],
                    capture_output=True, text=True, timeout=60)
            if cap.returncode == 0 and os.path.exists(png_path):
                report = validate_png(png_path, expected_lines=len(lines),
                                      bar_px=60)
                if report["ok"]:
                    return report, None
                failure = "image failed pixel validation: %s" % report["failed_checks"]
            else:
                failure = (cap.stderr or "").strip() or "screencapture failed"
        return None, failure + "\n" + PERMISSION_HELP
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if not keep_window:
            _osascript('tell application "Terminal" to close '
                       '(every window whose id is %s)' % win_id)


_DURATION = re.compile(r"in \d+\.\d+s")


def _normalise(text):
    """Ignore unittest's own wall-clock duration when comparing two runs."""
    return _DURATION.sub("in <duration>", text)


def _wait_until_idle(win_id, timeout):
    deadline = time.time() + timeout
    time.sleep(1.0)
    while time.time() < deadline:
        proc = _osascript(
            'tell application "Terminal" to get busy of selected tab '
            'of window id %s' % win_id)
        if proc.returncode != 0:
            return False
        if proc.stdout.strip() == "false":
            return True
        time.sleep(0.4)
    return False


def _sh(text):
    return "'" + str(text).replace("'", "'\\''") + "'"


def _as_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ----------------------------------------------------------------------
# 4. Path B -- headless Chrome rendering the verbatim text
# ----------------------------------------------------------------------

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title><style>
  html, body {{ margin: 0; padding: 0; background: #0b0f14; }}
  /* No box-shadow: its soft edge would defeat the exact bbox trim below. */
  .frame {{ display: inline-block; min-width: 900px;
            border-radius: 10px; overflow: hidden; }}
  .bar {{ background: #2b3038; height: 38px; display: flex; align-items: center;
          padding: 0 14px; gap: 8px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .t {{ flex: 1; text-align: center; color: #c6ccd4; font-size: 13px;
        font-family: -apple-system, system-ui, sans-serif; }}
  pre {{ margin: 0; background: {bg}; color: {fg}; padding: 22px 26px;
         font-family: Menlo, Monaco, "SF Mono", monospace;
         font-size: 15px; line-height: 1.45; white-space: pre; }}
</style></head><body>
<div class="frame">
  <div class="bar">
    <div class="dot" style="background:#ff5f57"></div>
    <div class="dot" style="background:#febc2e"></div>
    <div class="dot" style="background:#28c840"></div>
    <div class="t">{title}</div>
  </div>
  <pre>{body}</pre>
</div>
</body></html>
"""


def chrome_html_capture(shot, raw, png_path):
    if not os.path.exists(CHROME):
        return None, "Google Chrome is not installed at the expected path"

    lines = raw.splitlines() or [""]
    width = int(max(900, max(len(l) for l in lines) * 9.1 + 60))
    height = int(len(lines) * 21.8 + 140)

    page = HTML.format(
        title=html_mod.escape(shot.title),
        body=html_mod.escape(raw),          # verbatim, nothing added
        bg="#%02x%02x%02x" % SENTINEL_BG,
        fg="#%02x%02x%02x" % FOREGROUND,
    )
    tmp = tempfile.mkdtemp(prefix="lru-shot-")
    html_path = os.path.join(tmp, "page.html")
    with open(html_path, "w") as fh:
        fh.write(page)
    profile = tempfile.mkdtemp(prefix="lru-chrome-")
    try:
        last = ""
        # Chrome 153 writes the PNG and then hangs on shutdown (GCM /
        # updater / network service). So we do not wait for it to exit: we
        # poll for the screenshot to appear and settle, then kill it.
        for flag in ("--headless", "--headless=new"):
            cmd = [CHROME, flag, "--disable-gpu", "--no-first-run",
                   "--no-default-browser-check", "--hide-scrollbars",
                   "--disable-extensions", "--disable-sync",
                   "--user-data-dir=" + profile,
                   "--force-device-scale-factor=2",
                   "--window-size=%d,%d" % (width, height + 40),
                   "--screenshot=" + png_path,
                   "file://" + html_path]
            if os.path.exists(png_path):
                os.unlink(png_path)
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            try:
                if not _wait_for_stable_file(png_path, timeout=60):
                    last = "chrome produced no screenshot with %s" % flag
                    continue
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            _trim(png_path)
            report = validate_png(png_path, expected_lines=len(lines),
                                  bar_px=76)
            if report["ok"]:
                return report, None
            last = "render failed validation: %s" % report["failed_checks"]
        return None, last
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)


def _wait_for_stable_file(path, timeout, settle=1):
    """Wait for a file to appear and become a fully decodable image.

    A byte-size plateau is not a completion signal -- a writer that stalls
    mid-stream satisfies it and yields a half-written PNG. Decoding is.
    """
    deadline = time.time() + timeout
    stable, last_size = 0, -1
    while time.time() < deadline:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == last_size:
                stable += 1
                if stable > settle:
                    try:
                        with Image.open(path) as probe:
                            probe.load()
                        return True
                    except Exception:
                        stable = 0      # still being written
            else:
                stable = 0
            last_size = size
        time.sleep(0.5)
    return False


def _trim(path, margin=0):
    """Crop away the empty page area around the rendered window frame."""
    im = Image.open(path).convert("RGB")
    page = Image.new("RGB", im.size, (0x0B, 0x0F, 0x14))
    bbox = ImageChops.difference(im, page).getbbox()
    if not bbox:
        return
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(im.size[0], right + margin)
    bottom = min(im.size[1], bottom + margin)
    im.crop((left, top, right, bottom)).save(path)


# ----------------------------------------------------------------------
# 5. Path C -- Pillow draws the verbatim text. No browser, no permissions.
# ----------------------------------------------------------------------

def pillow_capture(shot, raw, png_path):
    scale = 2
    size = 15 * scale
    font = ImageFont.truetype(MENLO, size, index=0)
    bold = ImageFont.truetype(MENLO, size, index=1)
    lines = raw.splitlines() or [""]
    adv = font.getlength("M")
    line_h = int(size * 1.45)
    pad = 22 * scale
    bar = 38 * scale
    width = int(max(l and font.getlength(l) or 0 for l in lines) + pad * 2)
    width = max(width, int(adv * 84) + pad * 2)
    height = bar + pad * 2 + line_h * len(lines)

    im = Image.new("RGB", (width, height), SENTINEL_BG)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, width, bar], fill=(0x2B, 0x30, 0x38))
    for i, colour in enumerate(((0xFF, 0x5F, 0x57), (0xFE, 0xBC, 0x2E),
                                (0x28, 0xC8, 0x40))):
        cx = 18 * scale + i * 20 * scale
        r = 6 * scale
        draw.ellipse([cx - r, bar // 2 - r, cx + r, bar // 2 + r], fill=colour)
    tw = bold.getlength(shot.title)
    draw.text(((width - tw) / 2, bar / 2 - size * 0.65), shot.title,
              font=bold, fill=(0xC6, 0xCC, 0xD4))
    y = bar + pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill=FOREGROUND)
        y += line_h
    im.save(png_path)
    report = validate_png(png_path, expected_lines=len(lines), bar_px=bar)
    if report["ok"]:
        return report, None
    return None, "pillow render failed validation: %s" % report["failed_checks"]


# ----------------------------------------------------------------------

def _sha256_file(path):
    h = sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path():
    return os.path.join(PHOTO, "MANIFEST.json")


def _read_manifest():
    try:
        with open(_manifest_path()) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def _write_manifest(manifest):
    """Rewritten after every shot, so it always describes what is on disk."""
    tmp = _manifest_path() + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, _manifest_path())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html-only", action="store_true",
                        help="skip the Terminal capture (no focus stealing)")
    parser.add_argument("--keep-window", action="store_true",
                        help="leave the Terminal window open for debugging")
    args = parser.parse_args(argv)

    os.makedirs(PHOTO, exist_ok=True)
    use_terminal = not args.html_only
    terminal_skip_reason = "--html-only was passed"
    if use_terminal:
        ok, why = screen_recording_available()
        if not ok:
            use_terminal = False
            terminal_skip_reason = "Screen Recording unavailable: %s" % why
            print("Screen Recording permission is not available (%s)." % why)
            print("Falling back to rendering the verbatim captured output.")
            print("To get true Terminal screenshots instead, grant it under")
            print("  System Settings > Privacy & Security > Screen & System "
                  "Audio Recording")
            print("to the app running this script, relaunch it, and re-run "
                  "`make shots`.\n")
        else:
            print("NOTE: capturing real Terminal windows -- "
                  "do not type until done.\n")

    manifest = _read_manifest()
    for shot in SHOTS:
        print("[%s]" % shot.name)
        raw = run_and_capture(shot)
        print("  captured %d lines of real output" % len(raw.splitlines()))

        txt_path = os.path.join(PHOTO, shot.name + ".txt")
        final_png = os.path.join(PHOTO, shot.name + ".png")
        # Render into a scratch directory. The pair is only moved into photo/
        # once the image validates, so an aborted run can never leave a fresh
        # .txt beside a stale .png.
        staging = tempfile.mkdtemp(prefix="lru-stage-")
        png_path = os.path.join(staging, shot.name + ".png")
        report = method = None
        attempts = []
        if not use_terminal:
            attempts.append("terminal: skipped (%s)" % terminal_skip_reason)

        if use_terminal:
            report, why = try_terminal_capture(shot, raw, png_path, args.keep_window)
            if report:
                method = "terminal-screencapture"
            else:
                attempts.append("terminal: " + why)
                print("  terminal capture unavailable -- %s" % why.splitlines()[0])

        if report is None:
            report, why = chrome_html_capture(shot, raw, png_path)
            if report:
                method = "headless-chrome"
            else:
                attempts.append("chrome: " + why)
                print("  chrome render unavailable -- %s" % why.splitlines()[0])

        if report is None:
            report, why = pillow_capture(shot, raw, png_path)
            if report:
                method = "pillow-render"
            else:
                attempts.append("pillow: " + why)

        if report is None:
            shutil.rmtree(staging, ignore_errors=True)
            raise CaptureError("every capture path failed for %s:\n  %s"
                               % (shot.name, "\n  ".join(attempts)))

        # Commit both artefacts together, image first.
        os.replace(png_path, final_png)
        tmp_txt = txt_path + ".tmp"
        with open(tmp_txt, "w") as fh:
            fh.write(raw)            # verbatim, no header
        os.replace(tmp_txt, txt_path)
        shutil.rmtree(staging, ignore_errors=True)

        print("  image: %s via %s  %dx%d" % (os.path.basename(final_png), method,
                                             report["size"][0], report["size"][1]))
        entry = {
            "name": shot.name,
            "caption": shot.caption,
            "png": "photo/%s.png" % shot.name,
            "txt": "photo/%s.txt" % shot.name,
            "command": shot.title,
            "method": method,
            "stdout_sha256": sha256(raw.encode()).hexdigest(),
            "stdout_lines": len(raw.splitlines()),
            "png_sha256": _sha256_file(final_png),
            "validator": report,
            "paths_attempted": attempts or ["terminal (preferred, succeeded)"],
            "captured_at_utc": datetime.now(timezone.utc)
                               .replace(microsecond=0).isoformat(),
            "python": "%d.%d.%d" % sys.version_info[:3],
        }
        manifest = [m for m in manifest if m["name"] != shot.name] + [entry]
        manifest.sort(key=lambda m: m["name"])
        _write_manifest(manifest)

    print("\n%-28s %-24s %-12s %s" % ("SHOT", "METHOD", "SIZE", "BG MATCH"))
    print("-" * 76)
    for entry in manifest:
        print("%-28s %-24s %-12s %.2f"
              % (entry["name"], entry["method"],
                 "%dx%d" % tuple(entry["validator"]["size"]),
                 entry["validator"]["bg_match"]))
    print("-" * 76)
    print("Verify any shot against its real output, e.g.:")
    print("  python3 demo.py --section basic | diff - photo/01_assignment_example.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
