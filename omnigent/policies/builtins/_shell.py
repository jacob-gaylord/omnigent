"""Generic shell-command parsing shared by built-in shell-surface policies.

Built-in policies that gate the OS shell tool (``github`` for git/gh remote
operations, ``working_dir`` for directory / worktree switches) all face the
same problem: a single ``sys_os_shell`` ``command`` string can chain several
commands (``a && b ; c``), prefix them with env-assignments or wrappers
(``sudo``, ``env``, ``VAR=x``), and hide the real command inside a shell
interpreter (``bash -c "<cmd>"``) or ``eval``. A policy that only looked at
the first token would be trivially bypassable.

This module factors out the *generic* primitives for breaking a command into
its individual real invocations. It is deliberately policy-agnostic — it does
not know about git, directories, or any domain; each policy composes these
primitives with its own classification and decision logic (including its own
handling of un-tokenizable segments, which differs per policy).
"""

from __future__ import annotations

import re

# Leading tokens to skip when finding the real command in a segment — command
# wrappers that take the real command as their trailing arguments.
CMD_WRAPPERS: frozenset[str] = frozenset({"sudo", "env", "command", "time", "nohup", "exec"})

# Wrappers that take their own leading positional argument (a duration) — and
# optionally their own flags — *before* the real command: ``timeout 60 git …``,
# ``timeout --signal=KILL 5m git …``. Skipping only the wrapper word (as for
# ``CMD_WRAPPERS``) would leave the duration as the apparent command, so these
# need the extra positional consumed too.
CMD_WRAPPERS_WITH_DURATION: frozenset[str] = frozenset({"timeout"})

# ``timeout`` flags that consume a following value when written as two tokens
# (``-s KILL`` / ``--kill-after 10``). Used to avoid mistaking the flag's value
# for the duration positional. Combined ``--flag=value`` forms are a single
# token and need no special handling.
_DURATION_WRAPPER_VALUE_FLAGS: frozenset[str] = frozenset(
    {"-s", "--signal", "-k", "--kill-after"}
)

# Shell interpreters that run a command string passed via ``-c`` (or, for
# ``eval``, as positional words). Their inner command is parsed recursively so
# ``bash -c "git push …"`` is gated like a bare ``git push …`` rather than
# slipping past detection. Matched on the basename so ``/bin/bash`` counts too.
SHELL_INTERPRETERS: frozenset[str] = frozenset({"sh", "bash", "zsh", "dash", "ksh"})

# Guard against pathological nesting (``bash -c "bash -c …"``).
MAX_SHELL_NESTING = 4


def _extract_command_substitutions(command: str) -> tuple[str, list[str]]:
    """
    Pull ``$(...)`` and backtick command-substitution bodies out of a command.

    A substitution body is itself a command the shell *runs* (its output is
    interpolated), so ``x=$(git push …)`` executes the push even though the
    outer token looks like a plain env-assignment. To gate it, the body must be
    parsed as a command in its own right rather than swallowed by the assignment.

    :param command: The raw shell command string.
    :returns: ``(outer, bodies)`` — *outer* is *command* with each substitution
        replaced by a space (so the residue, e.g. ``x=``, parses harmlessly),
        and *bodies* is the list of inner command strings to parse separately.
        ``$(...)`` is matched with balanced-paren scanning so nested
        substitutions are captured; backticks are treated as non-nesting.
    """
    bodies: list[str] = []
    out: list[str] = []
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if ch == "$" and i + 1 < n and command[i + 1] == "(":
            depth, j = 1, i + 2
            while j < n and depth > 0:
                if command[j] == "(":
                    depth += 1
                elif command[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            bodies.append(command[i + 2 : j])
            out.append(" ")
            i = j + 1
            continue
        if ch == "`":
            j = command.find("`", i + 1)
            if j == -1:
                out.append(ch)
                i += 1
                continue
            bodies.append(command[i + 1 : j])
            out.append(" ")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), bodies


def split_command_segments(command: str) -> list[str]:
    """
    Split a shell command on chaining operators into individual segments.

    Splits on ``&&``, ``||``, ``;``, ``|``, a single ``&`` (background
    operator — ``a & b`` runs both) and newlines so that
    ``git add . && git push`` is evaluated as two segments. Command
    substitutions (``$(...)`` / backticks) are pulled out first and their
    bodies appended as their own segments, so a command hidden inside one is
    still gated. This is a naive split that does not honor operators appearing
    inside quotes — acceptable because the commands these policies gate do not
    embed these operators in quoted args in practice, and a mis-split only ever
    produces an extra segment (an unbalanced one surfaces as ASK, not a silent
    allow).

    :param command: The raw shell command string, e.g.
        ``"cd /repo && npm test"``.
    :returns: List of trimmed, non-empty segments, e.g.
        ``["cd /repo", "npm test"]``.
    """
    outer, bodies = _extract_command_substitutions(command)
    parts = re.split(r"&&|\|\||[;|\n&]", outer)
    segments = [seg.strip() for seg in parts if seg.strip()]
    for body in bodies:
        segments.extend(split_command_segments(body))
    return segments


def real_invocation_tokens(tokens: list[str]) -> list[str]:
    """
    Drop leading env-assignments and command wrappers to reach the real argv.

    :param tokens: shlex-split tokens of one segment, e.g.
        ``["sudo", "GIT_SSH=x", "git", "push"]``.
    :returns: Tokens starting at the real command (``["git", "push"]``), or
        empty when nothing remains.
    """
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in CMD_WRAPPERS or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
            index += 1
            continue
        if token in CMD_WRAPPERS_WITH_DURATION:
            index = _skip_duration_wrapper_args(tokens, index + 1)
            continue
        break
    return tokens[index:]


def _skip_duration_wrapper_args(tokens: list[str], index: int) -> int:
    """
    Skip a duration-wrapper's own flags and its leading duration positional.

    Given the index just past a ``CMD_WRAPPERS_WITH_DURATION`` word (e.g.
    ``timeout``), advance past any option flags it carries (consuming the value
    of a separate-token value flag such as ``-s KILL``) and then the single
    required duration positional, leaving *index* at the real command.

    :param tokens: The full token list of the segment.
    :param index: Index of the first token after the wrapper word.
    :returns: Index of the wrapped command's first token.
    """
    while index < len(tokens) and tokens[index].startswith("-"):
        flag = tokens[index]
        index += 1
        if flag in _DURATION_WRAPPER_VALUE_FLAGS and index < len(tokens):
            index += 1
    # The duration positional itself (``60`` / ``1.5s`` / ``5m``).
    if index < len(tokens):
        index += 1
    return index


def unwrap_shell_command(tokens: list[str]) -> str | None:
    """
    Return the inner command string of a shell-interpreter / ``eval`` wrapper.

    :param tokens: Real invocation tokens (env-prefixes / wrappers already
        stripped), e.g. ``["bash", "-c", "git push origin main"]`` or
        ``["eval", "git", "push"]``.
    :returns: The wrapped command string to re-parse, or ``None`` when *tokens*
        is not a shell-interpreter / ``eval`` invocation.
    """
    head = tokens[0].rsplit("/", 1)[-1]
    if head in SHELL_INTERPRETERS:
        for i, tok in enumerate(tokens):
            # ``-c`` takes the command string as its next argument. Match the
            # bare flag *and* combined short-flag bundles that include ``c``
            # (``-lc``, ``-ic``, ``-xc``): bash still reads the command from the
            # next operand, so ``bash -lc "git push …"`` must unwrap like
            # ``bash -c "git push …"`` rather than slip past as unrecognized.
            if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", tok) and i + 1 < len(tokens):
                return tokens[i + 1]
        return None
    if head == "eval":
        # ``eval`` runs its remaining words as a command (often a single quoted
        # string after shlex-splitting); rejoin them to re-parse.
        return " ".join(tokens[1:]) if len(tokens) > 1 else None
    return None
