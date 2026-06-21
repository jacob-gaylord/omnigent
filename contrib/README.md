# Custom additions

Additions on top of upstream omnigent, all running on **Claude + Codex
subscriptions only** (no API keys, no Databricks, no Pi).

## Agents (`examples/`)

### `PM` — Matt Pocock-method coding orchestrator
A polly-style orchestrator that runs the [Matt Pocock](https://aihero.dev)
engineering method: **align first** (grill → vertical-slice tickets, human
approves) → **domain-routed build** (frontend → Claude Code, backend → Codex,
each reviewed by the *other* vendor) → PR; the human merges. It also rotates to a fresh
thread (via `/handoff`) instead of letting context decay into the dumb zone.
Policies enforce human-controlled promotion (`blast_radius gate_pushes: true`)
and confine workers to their own worktree (`worktree_guard`).

```
omnigent run examples/pm
```

### `RnD` — two-headed brainstorming partner
A debby-style agent whose two heads run on Claude (`claude-sdk`) and Codex
(`codex-native`) — both on subscriptions, no OpenAI key — and shows both
perspectives side by side. Load the `debate` skill to have them critique each
other before converging.

```
omnigent run examples/rnd
```

## `contrib/omnigent-helper.py` — auto-title + fresh-thread nudge
A standalone companion (NOT part of the package, so it survives upgrades) that
watches the chat DB and:

1. **auto-titles** threads — replaces omnigent's raw first-message truncation
   with a real 3–6 word summary (one cheap `claude -p` call per new thread);
2. **fresh-over-compaction nudge** — when a thread's accumulated content passes
   ~100k tokens (Matt's "dumb zone"), prefixes its title with 🔴 as a "start a
   fresh thread" reminder (omnigent only auto-compacts at the full window, well
   past the point where a model gets dull).

Install as a systemd user service:

```bash
mkdir -p ~/.omnigent/tools ~/.config/systemd/user
cp contrib/omnigent-helper.py      ~/.omnigent/tools/
cp contrib/omnigent-helper.service ~/.config/systemd/user/
systemctl --user enable --now omnigent-helper.service
loginctl enable-linger "$USER"   # keep it running across reboots
```

Or run on a schedule with `python3 ~/.omnigent/tools/omnigent-helper.py --once` (cron).
