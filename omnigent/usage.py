"""Live subscription-usage for the web-UI usage panel.

Two providers, two sources — both local, both subscription-based:

* **claude** — parsed from ``claude -p "/usage"`` (the Claude Code CLI prints the
  current session + weekly limit usage for the logged-in subscription).
* **codex** — read from the most-recent Codex session rollout's ``rate_limits``
  event (``~/.codex/sessions/**/rollout-*.jsonl``). Codex receives 5h/weekly
  limits from the backend on every turn and records them there, so we need NO
  extra request — we just read the last snapshot.

Results are cached briefly so polling the panel doesn't spawn a ``claude`` call
on every tick.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
_TTL_SECONDS = 60.0


def _percent(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) if m else None


def _resets(line: str) -> str | None:
    m = re.search(r"resets\s+(.+?)(?:\s*\(|$)", line)
    return m.group(1).strip() if m else None


def get_claude_usage() -> dict[str, Any]:
    """Run ``claude -p "/usage"`` and parse the session + weekly limit lines."""
    try:
        proc = subprocess.run(
            ["claude", "-p", "/usage"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 — claude missing / timeout / etc.
        return {"available": False, "error": str(exc)}
    if proc.returncode != 0 and not proc.stdout:
        return {"available": False, "error": (proc.stderr or "claude usage failed").strip()[:200]}

    result: dict[str, Any] = {"available": True, "provider": "claude"}
    for line in proc.stdout.splitlines():
        low = line.lower()
        if "current session" in low:
            result["session"] = {"used_percent": _percent(line), "resets": _resets(line)}
        elif "current week (all models)" in low:
            result["week"] = {"used_percent": _percent(line), "resets": _resets(line)}
    if "session" not in result and "week" not in result:
        return {"available": False, "error": "could not parse claude usage output"}
    return result


def get_codex_usage() -> dict[str, Any]:
    """Read ``rate_limits`` from the most-recent Codex session rollout file."""
    base = Path(os.path.expanduser("~/.codex/sessions"))
    files = glob.glob(str(base / "**" / "rollout-*.jsonl"), recursive=True)
    if not files:
        return {"available": False, "error": "no codex sessions found"}
    latest = max(files, key=os.path.getmtime)

    rate_limits = None
    try:
        with open(latest, encoding="utf-8") as fh:
            for line in fh:
                if '"rate_limits"' not in line:
                    continue
                try:
                    payload = (json.loads(line).get("payload") or {})
                except json.JSONDecodeError:
                    continue
                if payload.get("rate_limits"):
                    rate_limits = payload["rate_limits"]  # keep the last (newest)
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    if not rate_limits:
        return {"available": False, "error": "no rate_limits in latest codex rollout"}

    out: dict[str, Any] = {
        "available": True,
        "provider": "codex",
        "plan_type": rate_limits.get("plan_type"),
    }
    for src, label in (("primary", "session"), ("secondary", "week")):
        window = rate_limits.get(src) or {}
        if window:
            out[label] = {
                "used_percent": window.get("used_percent"),
                "window_minutes": window.get("window_minutes"),
                "resets_at": window.get("resets_at"),
            }
    return out


def get_usage(force: bool = False) -> dict[str, Any]:
    """Combined Claude + Codex usage, cached for ``_TTL_SECONDS``."""
    now = time.time()
    cached = _CACHE["data"]
    if not force and cached and (now - _CACHE["ts"] < _TTL_SECONDS):
        return cached
    data = {
        "claude": get_claude_usage(),
        "codex": get_codex_usage(),
        "fetched_at": int(now),
    }
    _CACHE["data"] = data
    _CACHE["ts"] = now
    return data
