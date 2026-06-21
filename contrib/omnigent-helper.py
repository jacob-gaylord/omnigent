#!/usr/bin/env python3
"""omnigent-helper — a durable companion for the omnigent web UI.

Lives OUTSIDE the omnigent package (in ~/.omnigent/tools), so it survives
`uv tool upgrade omnigent`. It watches the chat DB and does two things omnigent
doesn't:

  1. AUTO-TITLE  — omnigent titles a thread with the raw first-message
     truncation. This replaces that with a real 3-6 word summary (one cheap
     `claude -p` call per thread, only the first time).

  2. FRESH-OVER-COMPACTION NUDGE — Matt Pocock's point: a model gets dumb as
     context grows (~100k tokens), and compaction keeps a *degraded* context
     alive. omnigent auto-compacts at the full window, well past the dumb zone.
     So when a thread's accumulated content crosses ~100k estimated tokens, we
     prefix its title with a red marker — a visible "start a fresh thread"
     reminder in the thread list. It does NOT delete or rotate anything; you
     stay in control.

Run modes:
  python3 omnigent-helper.py            # loop forever (default, every 60s)
  python3 omnigent-helper.py --once     # single pass (for cron)

Cron example (every 2 min):
  */2 * * * * $(command -v python3) ~/.omnigent/tools/omnigent-helper.py --once >> ~/.omnigent/tools/helper.log 2>&1
"""
import json
import os
import sqlite3
import subprocess
import sys
import time

DB = os.path.expanduser("~/.omnigent/chat.db")
POLL_SECONDS = 60
FRESH_TOKEN_LIMIT = 100_000        # Matt's dumb-zone threshold
FRESH_MARK = "\U0001F534 "          # 🔴 + space — title prefix = "start a fresh thread"
TITLE_MODEL = "claude-haiku-4-5"   # cheap model for titling; falls back to default if unknown
CHARS_PER_TOKEN = 4                # rough token estimate from content length


def _connect():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA busy_timeout=20000")
    return c


def _items(conn, conv_id):
    return list(
        conn.execute(
            "SELECT search_text FROM conversation_items "
            "WHERE conversation_id=? AND type='message' ORDER BY position ASC",
            (conv_id,),
        )
    )


def _generate_title(seed_text):
    prompt = (
        "Generate a concise 3-6 word title in Title Case for a conversation that "
        "begins with the text below. Output ONLY the title — no quotes, no "
        "trailing punctuation, no preamble.\n\n" + seed_text[:1500]
    )
    for args in (["claude", "-p", "--model", TITLE_MODEL, prompt], ["claude", "-p", prompt]):
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=90)
            if out.returncode != 0:
                continue
            line = (out.stdout or "").strip().splitlines()
            title = line[0].strip().strip('"').strip("'").rstrip(".") if line else ""
            if title:
                return title[:80]
        except Exception:
            continue
    return None


def _is_raw_autotitle(base, first_text):
    """True if `base` looks like omnigent's raw first-message truncation."""
    b = base.rstrip("… ").strip()  # strip a trailing ellipsis
    if not b:
        return True
    # raw titles are a short prefix of the opening message
    return len(b) <= 65 and first_text[: len(b)].lower() == b.lower()


def pass_once(verbose=True):
    conn = _connect()
    convs = list(
        conn.execute(
            "SELECT id, COALESCE(title, ''), COALESCE(session_usage, '') "
            "FROM conversations WHERE kind='default'"
        )
    )
    changed = 0
    for conv_id, title, _usage in convs:
        items = _items(conn, conv_id)
        if not items:
            continue
        first_text = (items[0][0] or "").strip()
        est_tokens = sum(len(r[0] or "") for r in items) // CHARS_PER_TOKEN

        has_mark = title.startswith(FRESH_MARK)
        base = title[len(FRESH_MARK):] if has_mark else title

        new_base = base
        if _is_raw_autotitle(base, first_text) and first_text:
            generated = _generate_title(first_text)
            if generated:
                new_base = generated

        need_mark = est_tokens >= FRESH_TOKEN_LIMIT
        new_title = (FRESH_MARK if need_mark else "") + new_base

        if new_title and new_title != title:
            conn.execute(
                "UPDATE conversations SET title=?, updated_at=updated_at WHERE id=?",
                (new_title, conv_id),
            )
            conn.commit()
            changed += 1
            if verbose:
                tag = "  (~%dk, FRESH)" % (est_tokens // 1000) if need_mark else ""
                print(f"[titled] {conv_id[:12]}  -> {new_title!r}{tag}", flush=True)
    conn.close()
    return changed


def main():
    once = "--once" in sys.argv
    if once:
        n = pass_once()
        print(f"done: {n} thread(s) updated", flush=True)
        return
    print("omnigent-helper running (Ctrl-C to stop) — titling + fresh-thread nudges", flush=True)
    while True:
        try:
            pass_once(verbose=True)
        except Exception as exc:  # never die on a transient error
            print(f"[warn] {type(exc).__name__}: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
