"""AntigravityExecutor: run agents using Google's Antigravity SDK.

Wraps the ``google-antigravity`` SDK as the agent runtime, the SDK-wrap
counterpart to :class:`~omnigent.inner.openai_agents_sdk_executor.OpenAIAgentsSDKExecutor`
and :class:`~omnigent.inner.claude_sdk_executor.ClaudeSDKExecutor`: a direct
in-process Executor with ``handles_tools_internally() -> True``, per-session
agent reuse, and a streaming :meth:`run_turn` mapping SDK events onto Omnigent
:class:`~omnigent.inner.executor.ExecutorEvent` instances.

.. note::
   The SDK is not pure-Python: it launches a bundled native ``localharness``
   binary linked against a recent glibc (needs ``GLIBC_ABI_DT_RELR``), so the
   harness only runs on hosts with glibc ≳ 2.36; older hosts surface an
   :class:`ExecutorError`.

Default model is Gemini 3.5 Flash; the SDK can also drive Claude / GPT-OSS.

Streaming model:

- ``agent.conversation.receive_steps()`` yields :class:`Step` objects as the
  turn runs (``content_delta``, ``thinking_delta``, ``tool_calls``,
  ``status``, ``usage_metadata``), ending once the turn goes idle. A producer
  task drives ``send()`` + ``receive_steps()`` and feeds a per-turn
  :class:`asyncio.Queue`; :meth:`run_turn` drains it and yields mapped events.
- Tool *requests* derive from ``Step.tool_calls`` (deduped by call id); tool
  *completions* (with payload, error, duration) arrive via a registered
  ``PostToolCallHook`` and pair back by call id.
- Cancellation: :meth:`interrupt_session` -> ``conversation.cancel()``, which
  surfaces ``TurnCancelled``.

Policy enforcement mirrors the peer SDK executors so operator guardrails apply
to this runtime too:

- TOOL_CALL phase: a registered ``PreToolCallDecideHook`` consults the
  executor's ``_policy_evaluator`` BEFORE each tool runs. A DENY verdict blocks
  the call (the SDK rejects it without executing the tool — chiefly its bundled
  native tools; bridged Omnigent tools are skipped here because they are already
  TOOL_CALL-gated server-side on the dispatch path, see
  :meth:`_build_pre_tool_hook`); an ASK verdict routes through the executor's
  ``_elicitation_handler`` (deny when none is wired). The bridged skip set is
  bound per-agent into the hook closure (not stored on the executor) and a
  bundled native tool is gated even if its wire name collides with a bridged
  name, so a colliding name can't bypass policy. This is the ``can_use_tool``
  gate of :class:`~omnigent.inner.claude_sdk_executor.ClaudeSDKExecutor`,
  expressed through the SDK's decide-hook surface.
- LLM_REQUEST / LLM_RESPONSE phases: :meth:`run_turn` evaluates these around
  the model call (deny-before-spawn / deny-after-stream), matching the peer
  executors so prompt-deny and turn-completion-block policies are honored. As
  in the peers, response text streams live, so an LLM_RESPONSE DENY blocks
  TurnComplete (and persistence) but does not un-send already-emitted deltas;
  see the note at the LLM_RESPONSE block in :meth:`run_turn`.

Both ``_policy_evaluator`` and ``_elicitation_handler`` are wired in by the
harness ExecutorAdapter; every phase is a no-op when the evaluator is absent.

Authentication is Gemini-native: a direct API key (``api_key``) or Vertex AI
(``vertex`` + ``project`` + ``location``). The SDK has no OpenAI-compatible
``base_url``, so there is deliberately no gateway / Databricks routing path.

SDK touchpoints are isolated in :meth:`_open_agent`, :meth:`_build_sdk_tools`,
:meth:`_build_post_tool_hook`, :meth:`_build_pre_tool_hook`, and
:meth:`_drive_turn`, and duck-typed to tolerate drift across the v0.1.x
surface. Unit tests stub the SDK module.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, TypeAlias

from omnigent.spec.types import RetryPolicy

from .executor import (
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    ReasoningChunk,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    ToolCallStatus,
    ToolSpec,
    TurnCancelled,
    TurnComplete,
    classify_tool_result,
)

logger = logging.getLogger(__name__)

# Default model when neither spec nor provider pins one.
_ANTIGRAVITY_DEFAULT_MODEL = "gemini-3.5-flash"

# Wire names of the SDK's bundled native tools (``BuiltinTools`` enum values),
# used as the fallback "is this a native tool?" set when the live enum can't be
# resolved (e.g. the SDK isn't importable, or a test stub omits it). The live
# enum is preferred at runtime (see :func:`_native_tool_wire_names`); this
# mirror only guards drift / absence. Kept in sync with
# ``google.antigravity.types.BuiltinTools``.
_NATIVE_TOOL_WIRE_NAMES: frozenset[str] = frozenset(
    {
        "list_directory",
        "search_directory",
        "find_file",
        "view_file",
        "create_file",
        "edit_file",
        "run_command",
        "ask_question",
        "start_subagent",
        "generate_image",
        "finish",
    }
)

# Sentinel the producer pushes when the step stream is exhausted, so the
# consumer can distinguish "turn finished" from a queued event.
_STREAM_DONE = object()


class _NeverRaisedError(BaseException):
    """Never-raised cancellation sentinel used when the SDK's type is
    unavailable, so the guarding ``except`` clause matches nothing.
    """


# SDK objects treated as opaque and duck-typed below. Kept as ``Any`` so
# ``google-antigravity`` stays an optional import at type-check time.
SDKAgent: TypeAlias = Any  # type: ignore[explicit-any]
SDKConversation: TypeAlias = Any  # type: ignore[explicit-any]
SDKStep: TypeAlias = Any  # type: ignore[explicit-any]
SDKToolCall: TypeAlias = Any  # type: ignore[explicit-any]
SDKToolResult: TypeAlias = Any  # type: ignore[explicit-any]
SDKUsage: TypeAlias = Any  # type: ignore[explicit-any]
SDKTool: TypeAlias = Any  # type: ignore[explicit-any]
SDKHook: TypeAlias = Any  # type: ignore[explicit-any]
SDKHookResult: TypeAlias = Any  # type: ignore[explicit-any]
SDKConfig: TypeAlias = Any  # type: ignore[explicit-any]
SDKHookContext: TypeAlias = Any  # type: ignore[explicit-any]
ToolArgs: TypeAlias = dict[str, Any]  # type: ignore[explicit-any]
ToolResult: TypeAlias = Any  # type: ignore[explicit-any]
# Aliased so the explicit ``Any`` (with its ``type: ignore``) lives here, not
# scattered across multi-line signatures where reformatting would split them.
_StrAnyDict: TypeAlias = dict[str, Any]  # type: ignore[explicit-any]
_EventQueue: TypeAlias = asyncio.Queue[Any]  # type: ignore[explicit-any]
_ToolCallable: TypeAlias = Callable[..., Awaitable[ToolResult]]  # type: ignore[explicit-any]

# Tool-execution callback the harness ExecutorAdapter wires in (assigns
# ``executor._tool_executor`` when unset). Routes a tool call ``(name, args)``
# back through the Session registry, so the in-SDK agent reaches Omnigent's
# sys / sub-agent / MCP tools under policy.
ToolExecutor: TypeAlias = Callable[  # type: ignore[explicit-any]
    [str, dict[str, Any]], Awaitable[dict[str, Any]]
]

# Elicitation handler the harness ExecutorAdapter wires in. Returns whether
# the user approves a tool call. Kept SDK-agnostic (a plain async callable) so
# this module need not import the adapter's verdict / elicitation types —
# mirrors :data:`omnigent.inner.claude_sdk_executor.ElicitationHandler`.
ElicitationHandler: TypeAlias = Callable[  # type: ignore[explicit-any]
    [str, dict[str, Any]], Awaitable[bool]
]

# Policy-evaluation callback the harness ExecutorAdapter wires in (assigns
# ``executor._policy_evaluator``). Called with a proto-style phase string
# (``"PHASE_TOOL_CALL"`` / ``"PHASE_LLM_REQUEST"`` / ``"PHASE_LLM_RESPONSE"``)
# and an event-data dict; returns a verdict object exposing ``.action`` (e.g.
# ``"POLICY_ACTION_DENY"``) and ``.reason``. Typed structurally as ``Any`` so
# this module stays decoupled from the adapter's ``PolicyVerdictPayload``.
PolicyEvaluator: TypeAlias = Callable[  # type: ignore[explicit-any]
    [str, dict[str, Any]], Awaitable[Any]
]


def _ensure_antigravity_sdk() -> ModuleType:
    """Import and return the ``google.antigravity`` module.

    :returns: The imported ``google.antigravity`` module.
    :raises ImportError: If ``google-antigravity`` isn't installed — surfaced
        on the first :meth:`run_turn` so it's a request-time error, not an
        app-boot crash.
    """
    try:
        # importlib keeps the ``-> ModuleType`` return type clean regardless of
        # whether the optional extra is installed at mypy time (a static import
        # would resolve to ``Any`` and trip ``warn_return_any``).
        return importlib.import_module("google.antigravity")
    except ImportError as exc:
        raise ImportError(
            "AntigravityExecutor requires the 'google-antigravity' package. "
            "Install it with: pip install google-antigravity (or "
            "pip install 'omnigent[antigravity]')."
        ) from exc


def _latest_user_text(messages: list[Message]) -> str:
    """Extract the newest user-authored text to feed the agent's next turn.

    The SDK keeps its own per-agent conversation state, so each turn only
    needs the latest user input. Concatenates the text parts of the last
    ``user`` message; falls back to the last message of any role.

    :param messages: The Omnigent turn message list (role / content dicts),
        e.g. ``[{"role": "user", "content": "review this PR"}]``.
    :returns: The user input text for this turn, or ``""`` when none.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        return _content_to_text(message.get("content"))
    if messages:
        return _content_to_text(messages[-1].get("content"))
    return ""


def _content_to_text(content: Any) -> str:  # type: ignore[explicit-any]
    """Flatten a message ``content`` value to plain text.

    Handles a bare string, a list of content blocks
    (``{"type": "text"|"input_text", "text": ...}``), or any other JSON value
    (serialized as a last resort).

    :param content: The message ``content`` field — a ``str``, a ``list`` of
        block dicts, or any other JSON-serializable value.
    :returns: A plain-text rendering of *content*.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return json.dumps(content)


def _tool_name(raw_name: Any) -> str:  # type: ignore[explicit-any]
    """Normalize an SDK tool name to a plain string.

    ``ToolCall.name`` / ``ToolResult.name`` may be a ``BuiltinTools`` enum or
    a bare string; ``.value`` (enum) or ``str`` both yield the wire name.

    :param raw_name: The SDK-supplied tool name (enum member or ``str``).
    :returns: The tool's wire name, e.g. ``"sys_shell"``, or ``""`` when the
        name is missing / not string-like.
    """
    name = getattr(raw_name, "value", raw_name)
    return name if isinstance(name, str) and name else ""


def _enum_name(value: Any) -> str:  # type: ignore[explicit-any]
    """Return an enum member's ``name`` (e.g. ``"TOOL_CALL"``), or ``""``.

    Lets the executor compare ``Step.type`` / ``Step.status`` by stable name
    string, tolerating enum-member drift without importing the enum types.

    :param value: An enum member such as ``StepType.TOOL_CALL`` or ``None``.
    :returns: The member's ``name`` attribute, or ``""`` when absent.
    """
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else ""


def _native_tool_wire_names(antigravity: ModuleType) -> frozenset[str]:
    """Resolve the wire names of the SDK's bundled native tools.

    Reads ``google.antigravity.types.BuiltinTools`` member ``.value``\\ s when
    present, falling back to :data:`_NATIVE_TOOL_WIRE_NAMES` when the enum is
    missing (e.g. a test stub) or unreadable. Used by the pre-tool decide hook
    to keep native tools policy-gated even when a native tool's wire name
    collides with a bridged tool name.

    :param antigravity: The imported ``google.antigravity`` module (the fake
        SDK module under test).
    :returns: The native tool wire names, or the static fallback set.
    """
    builtin = getattr(getattr(antigravity, "types", None), "BuiltinTools", None)
    if builtin is None:
        return _NATIVE_TOOL_WIRE_NAMES
    try:
        names = {member.value for member in builtin if isinstance(member.value, str)}
    except TypeError:
        # Not iterable as an enum — fall back to the static mirror.
        return _NATIVE_TOOL_WIRE_NAMES
    return frozenset(names) or _NATIVE_TOOL_WIRE_NAMES


def _is_native_tool_call(data: SDKToolCall, native_wire_names: frozenset[str]) -> bool:
    """Whether a pending ``ToolCall`` is one of the SDK's bundled native tools.

    Two independent signals, either of which marks the call native:

    - **Enum-typed name** — ``ToolCall.name`` is a ``BuiltinTools`` enum member
      for native tools and a plain ``str`` for bridged / custom callables
      (``name: BuiltinTools | str`` in the SDK). The enum check is the precise
      discriminator: a native call is native even if its wire name equals a
      bridged tool's name.
    - **Native wire name** — the resolved wire name is in *native_wire_names*.
      Covers SDKs (and test stubs) that pass native names as plain strings.

    Either signal forces the call to be policy-gated rather than skipped as a
    bridged tool, closing the name-collision bypass.

    :param data: The SDK ``ToolCall`` presented to the pre-tool hook.
    :param native_wire_names: The native tool wire names for this agent's SDK.
    :returns: ``True`` when the call targets a bundled native tool.
    """
    raw_name = getattr(data, "name", None)
    # A ``BuiltinTools`` enum member exposes ``.value`` (the wire name) and is
    # not a bare ``str``; bridged callables always present a bare ``str`` name.
    if not isinstance(raw_name, str) and isinstance(getattr(raw_name, "value", None), str):
        return True
    return _tool_name(raw_name) in native_wire_names


@dataclass
class _PendingTool:
    """A tool call awaiting its completion event.

    :param name: The tool's wire name, e.g. ``"sys_shell"``.
    :param started: ``time.monotonic()`` at :class:`ToolCallRequest` emit,
        used to compute ``ToolCallComplete.duration_ms``.
    """

    name: str
    started: float


@dataclass
class _AntigravitySessionState:
    """Per-session state for the Antigravity executor.

    Mutated in place (not replaced) across turns so the agent's
    ``PostToolCallHook``, which closes over this object, always sees the
    current turn's queue and pending-tool table.

    :param agent: Cached SDK ``Agent`` reused across turns to persist the
        SDK's conversation state, or ``None`` before the first turn.
    :param conversation: The agent's live ``Conversation`` (from
        ``agent.conversation``) — source of the ``Step`` stream and the
        ``cancel()`` entry point. ``None`` until the agent is opened.
    :param agent_signature: ``(model, system_prompt, tool_signature)`` key; a
        change forces an agent rebuild. A model change thus resets the SDK
        conversation — fine since mid-session model switches are rare.
    :param pending_tools: Open tool calls keyed by call id, populated on
        :class:`ToolCallRequest` and drained by the ``PostToolCallHook``.
    :param active_queue: The current turn's event queue (the
        ``PostToolCallHook`` enqueues completions here), or ``None`` between
        turns.
    :param last_usage: Most recent ``UsageMetadata`` this turn, surfaced on
        the terminal :class:`TurnComplete`.
    :param interrupt_requested: Set by :meth:`interrupt_session` so the
        consumer suppresses a trailing :class:`TurnComplete` after cancel.
    """

    agent: SDKAgent = None
    conversation: SDKConversation = None
    agent_signature: tuple[str, str, str] | None = field(default=None)
    pending_tools: dict[str, _PendingTool] = field(default_factory=dict)
    active_queue: _EventQueue | None = field(default=None)
    last_usage: SDKUsage = None
    interrupt_requested: bool = False


class AntigravityExecutor(Executor):
    """Execute turns using the Google Antigravity SDK."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        vertex: bool = False,
        project: str | None = None,
        location: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Create an AntigravityExecutor.

        :param model: Default model when per-turn :attr:`ExecutorConfig.model`
            is unset, e.g. ``"gemini-3.5-flash"`` (from
            ``HARNESS_ANTIGRAVITY_MODEL``). ``None`` falls back to
            :data:`_ANTIGRAVITY_DEFAULT_MODEL`.
        :param api_key: Direct Antigravity / Gemini API key (from
            ``HARNESS_ANTIGRAVITY_API_KEY``). ``None`` lets the SDK read its
            ambient ``GEMINI_API_KEY`` / ``ANTIGRAVITY_API_KEY``.
        :param vertex: When ``True``, authenticate via Vertex AI (GCP ADC)
            instead of an API key. Requires *project* / *location*.
        :param project: GCP project id for the Vertex AI path, e.g.
            ``"my-gcp-project"``. ``None`` unless *vertex* is set.
        :param location: GCP region for the Vertex AI path, e.g.
            ``"us-central1"``. ``None`` unless *vertex* is set.
        :param retry_policy: Optional retry policy, reserved for parity with
            the other SDK executors. ``None`` uses defaults.
        """
        self._model_override = model
        self._api_key = api_key
        self._vertex = vertex
        self._project = project
        self._location = location
        self._retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self._session_states: dict[str, _AntigravitySessionState] = {}
        # Set by the harness ExecutorAdapter when unset; the SDK tools route
        # invocations through this to reach Omnigent's tools under policy.
        self._tool_executor: ToolExecutor | None = None
        # Elicitation handler wired in by ExecutorAdapter. When set, a
        # TOOL_CALL-phase ASK verdict is resolved by an async approve/deny
        # round-trip through the Omnigent elicitation system; ``None`` until
        # the adapter installs it (an ASK then fails closed → deny).
        self._elicitation_handler: ElicitationHandler | None = None
        # Policy evaluator wired in by ExecutorAdapter (assigned to
        # ``_policy_evaluator`` when unset). Accessed via ``getattr`` at use
        # sites — same pattern as the peer SDK executors — so every policy
        # phase is a no-op until the adapter installs it. Declared here for
        # discoverability; the adapter owns assignment.
        self._policy_evaluator: PolicyEvaluator | None = None

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def handles_tools_internally(self) -> bool:
        # The SDK runs its own agentic loop and executes its own tools, so the
        # Session must not re-execute on ToolCallRequest — they're informational.
        return True

    def supports_tool_boundary_interrupt(self) -> bool:
        # interrupt_session() -> conversation.cancel() stops the loop at the
        # next safe boundary, so queued input applies after interrupt.
        return True

    def max_context_tokens(self) -> int | None:
        return None

    def _session_key(self, messages: list[Message]) -> str:
        """Resolve the per-session key from the turn's trailing message.

        :param messages: The Omnigent turn message list; the last entry's
            ``session_id`` (top-level or under ``metadata``) is the key.
        :returns: The session id string, or ``"default"`` when none is set.
        """
        if messages:
            last = messages[-1]
            if last.get("session_id"):
                return str(last["session_id"])
            metadata = last.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("session_id"):
                return str(metadata["session_id"])
        return "default"

    @staticmethod
    def _tool_signature(tools: list[ToolSpec]) -> str:
        """Stable cache key for a tool set (names only — enough to detect change).

        :param tools: Omnigent tool specs for the turn.
        :returns: Deterministic JSON string of the sorted tool names.
        """
        names = sorted(str(tool.get("name", "")) for tool in tools)
        return json.dumps(names, separators=(",", ":"))

    async def close_session(self, session_key: str) -> None:
        """Close and drop the SDK agent for *session_key*, if any.

        :param session_key: The Omnigent session id whose agent to release.
        """
        state = self._session_states.pop(session_key, None)
        if state is not None and state.agent is not None:
            await self._close_agent(state.agent)

    async def close(self) -> None:
        """Close every live SDK agent and clear all session state."""
        for state in list(self._session_states.values()):
            if state.agent is not None:
                await self._close_agent(state.agent)
        self._session_states.clear()

    async def interrupt_session(self, session_key: str) -> bool:
        """Interrupt the in-flight turn for *session_key* via the SDK.

        Marks the session interrupted (suppressing a trailing
        :class:`TurnComplete`) and asks the SDK to cancel. ``receive_steps()``
        then raises ``AntigravityCancelledError`` or yields a ``CANCELED``
        step, which the consumer surfaces as :class:`TurnCancelled`.

        :param session_key: The Omnigent session id to interrupt.
        :returns: ``True`` if a live conversation was asked to cancel,
            ``False`` when the session has no open conversation.
        """
        state = self._session_states.get(session_key)
        if state is None or state.conversation is None:
            return False
        state.interrupt_requested = True
        with contextlib.suppress(Exception):
            await state.conversation.cancel()
        return True

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """Run one turn through the Antigravity SDK, streaming events live.

        Spawns a producer task that drives the SDK turn and feeds a per-turn
        queue, then drains it yielding mapped events (text / reasoning / tool
        request / tool completion) and a terminal :class:`TurnComplete` —
        or :class:`TurnCancelled` on interrupt, :class:`ExecutorError` on
        failure.

        :param messages: The conversation messages for this turn.
        :param tools: Omnigent tool specs exposed to the agent as callables.
        :param system_prompt: The agent's system instructions.
        :param config: Per-turn config; ``config.model`` (e.g. from the REPL
            ``/model`` command) wins over the constructor default.
        :yields: :class:`TextChunk`, :class:`ReasoningChunk`,
            :class:`ToolCallRequest`, :class:`ToolCallComplete`, and a
            terminal :class:`TurnComplete` / :class:`TurnCancelled` /
            :class:`ExecutorError`.
        """
        model = (
            (config.model if config and config.model else None)
            or self._model_override
            or (_ANTIGRAVITY_DEFAULT_MODEL)
        )
        session_key = self._session_key(messages)
        prompt = _latest_user_text(messages)

        # ── LLM_REQUEST policy evaluation ────────────────────────
        # Mirror the peer executors: when the adapter wired a policy evaluator,
        # evaluate the request before the model call (the producer's
        # ``conversation.send()``) so a DENY aborts the turn without spawning it.
        policy_eval = getattr(self, "_policy_evaluator", None)
        if policy_eval is not None:
            req_data: _StrAnyDict = {
                "model": model,
                "messages_count": len(messages),
                "tools_count": len(tools),
                "system_prompt_preview": (system_prompt[:200] if system_prompt else ""),
                "last_user_message": prompt[:500],
            }
            req_verdict = await policy_eval("PHASE_LLM_REQUEST", req_data)
            if getattr(req_verdict, "action", None) == "POLICY_ACTION_DENY":
                deny_reason = getattr(req_verdict, "reason", None) or "no reason given"
                yield ExecutorError(message=f"LLM call denied by policy: {deny_reason}")
                return

        try:
            state = await self._ensure_agent(
                session_key,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )
        except ImportError as exc:
            yield ExecutorError(message=str(exc), retryable=False)
            return
        except Exception as exc:
            logger.exception("Antigravity agent construction failed")
            yield ExecutorError(message=f"Antigravity agent setup failed: {exc}", retryable=False)
            return

        event_queue: _EventQueue = asyncio.Queue()
        state.active_queue = event_queue
        state.pending_tools.clear()
        state.last_usage = None
        state.interrupt_requested = False

        producer = asyncio.create_task(
            self._drive_turn(state, prompt),
            name=f"antigravity-turn:{session_key}",
        )
        final_text_parts: list[str] = []
        tool_calls_count = 0
        cancelled = False
        errored = False
        try:
            while True:
                item = await event_queue.get()
                if item is _STREAM_DONE:
                    break
                if isinstance(item, TurnCancelled):
                    cancelled = True
                    yield item
                    continue
                if isinstance(item, ExecutorError):
                    errored = True
                    yield item
                    continue
                if isinstance(item, TextChunk):
                    final_text_parts.append(item.text)
                elif isinstance(item, ToolCallRequest):
                    tool_calls_count += 1
                yield item
        finally:
            state.active_queue = None
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

        if errored or cancelled:
            return
        if state.interrupt_requested:
            # Cancel was requested but the SDK ended the turn cleanly, so the
            # producer never emitted TurnCancelled. Emit it here instead of a
            # TurnComplete for a turn the user interrupted.
            yield TurnCancelled(reason="user_cancelled", phase="model")
            return
        usage = self._extract_usage(state.last_usage)
        response_text = "".join(final_text_parts) or None

        # ── LLM_RESPONSE policy evaluation ───────────────────────
        # Mirror the peer executors: evaluate after the stream completes but
        # before TurnComplete so a DENY blocks the turn from completing and the
        # response from being persisted as a finished assistant message.
        #
        # Known cross-executor limitation (NOT antigravity-specific): text
        # deltas (TextChunk) have already been yielded live during streaming
        # above, so an LLM_RESPONSE DENY can suppress TurnComplete but cannot
        # un-send deltas a client already received. This is the SAME shape as
        # the peer SDK executors — ClaudeSDKExecutor.run_turn yields TextChunk
        # per ``content_block_delta`` / per ``TextBlock`` and only evaluates
        # PHASE_LLM_RESPONSE after the stream ends (see its "LLM_RESPONSE policy
        # evaluation" block), and OpenAIAgentsSDKExecutor yields TextChunk per
        # ``response.output_text.delta`` and does not gate the response phase at
        # all. Buffering deltas until the verdict would be a true output-block
        # gate, but it must be decided once across all streaming executors (it
        # trades away live streaming UX), so it is deliberately NOT diverged
        # here. The post-stream DENY still prevents the blocked turn from being
        # recorded / acted on, matching claude-sdk.
        if policy_eval is not None:
            resp_data: _StrAnyDict = {
                "model": model,
                "text_preview": (response_text[:500] if response_text else ""),
                "tool_calls_count": tool_calls_count,
            }
            if usage is not None:
                resp_data["usage"] = usage
            resp_verdict = await policy_eval("PHASE_LLM_RESPONSE", resp_data)
            if getattr(resp_verdict, "action", None) == "POLICY_ACTION_DENY":
                deny_reason = getattr(resp_verdict, "reason", None) or "no reason given"
                yield ExecutorError(message=f"LLM response denied by policy: {deny_reason}")
                return

        yield TurnComplete(response=response_text, usage=usage)

    # ── SDK touchpoints (isolated; duck-typed; verified against v0.1.x) ──

    async def _drive_turn(self, state: _AntigravitySessionState, prompt: str) -> None:
        """Producer: drive one SDK turn, enqueuing mapped events.

        Sends *prompt*, then iterates ``receive_steps()`` mapping each
        :class:`Step` to text / reasoning / tool-request events on
        ``state.active_queue``. Tool *completions* are enqueued separately by
        the ``PostToolCallHook`` on the same event loop. Always enqueues
        :data:`_STREAM_DONE` last so the consumer stops cleanly.

        :param state: The session state (conversation, queue, pending tools).
        :param prompt: The user text to send for this turn.
        """
        queue = state.active_queue
        assert queue is not None  # set by run_turn before spawning this task
        conversation = state.conversation
        seen_tool_ids: set[str] = set()
        try:
            await conversation.send(prompt)
            async for step in conversation.receive_steps():
                # Only surface MODEL->USER steps. The SDK echoes the user's
                # input and environment-directed steps in the same stream (its
                # ``receive_chunks`` filters identically); without this the
                # user's prompt leaks back into the assistant response.
                is_model_to_user = (
                    _enum_name(getattr(step, "source", None)) == "MODEL"
                    and _enum_name(getattr(step, "target", None)) == "USER"
                )
                if is_model_to_user:
                    thinking_delta = getattr(step, "thinking_delta", "") or ""
                    if thinking_delta:
                        queue.put_nowait(
                            ReasoningChunk(delta=thinking_delta, event_type="reasoning_text")
                        )
                    content_delta = getattr(step, "content_delta", "") or ""
                    if content_delta:
                        queue.put_nowait(TextChunk(text=content_delta))

                usage = getattr(step, "usage_metadata", None)
                if usage is not None:
                    state.last_usage = usage

                self._emit_tool_requests(step, state, seen_tool_ids)

                step_type = _enum_name(getattr(step, "type", None))
                status = _enum_name(getattr(step, "status", None))

                # Fallback completion: results normally arrive via the
                # PostToolCallHook, which pops the call from pending_tools. If a
                # TOOL_CALL step reaches a terminal status with the call still
                # pending (e.g. an error surfaced outside the hook), close it
                # from the step. Both paths pop, so it fires once.
                if step_type == "TOOL_CALL" and status in (
                    "DONE",
                    "ERROR",
                    "TERMINAL_ERROR",
                    "CANCELED",
                ):
                    self._complete_pending_from_step(step, state, status)

                if status == "CANCELED":
                    queue.put_nowait(TurnCancelled(reason="user_cancelled", phase="model"))
                    return
                if status in ("ERROR", "TERMINAL_ERROR") and step_type != "TOOL_CALL":
                    # Turn-level failure (tool errors close above). TERMINAL_ERROR
                    # is non-retryable; plain ERROR may succeed on retry. Fall back
                    # to a generic message when the SDK reports none, so the turn
                    # isn't mis-reported as a silent empty success.
                    message = (
                        getattr(step, "error", "") or f"Antigravity turn failed (status={status})"
                    )
                    queue.put_nowait(ExecutorError(message=message, retryable=(status == "ERROR")))
                    return
        except asyncio.CancelledError:
            # Consumer teardown; conversation.cancel() already requested by
            # interrupt_session, nothing to emit.
            raise
        except self._cancelled_error_type():
            queue.put_nowait(TurnCancelled(reason="user_cancelled", phase="model"))
        except Exception as exc:
            logger.exception("Antigravity turn failed")
            queue.put_nowait(
                ExecutorError(message=f"Antigravity turn failed: {exc}", retryable=True)
            )
        finally:
            queue.put_nowait(_STREAM_DONE)

    def _emit_tool_requests(
        self,
        step: SDKStep,
        state: _AntigravitySessionState,
        seen_tool_ids: set[str],
    ) -> None:
        """Enqueue a :class:`ToolCallRequest` for each new tool call in *step*.

        Calls repeat across step transitions, so each call id is emitted once
        and its start time recorded for the matching :class:`ToolCallComplete`.
        Id-less calls are always emitted (can't dedupe) and get a synthetic id
        so the completion hook can still pair them.

        :param step: The current SDK :class:`Step`.
        :param state: The session state (records pending tools by call id).
        :param seen_tool_ids: Call ids already emitted this turn (mutated).
        """
        queue = state.active_queue
        if queue is None:
            return
        for call in getattr(step, "tool_calls", None) or []:
            name = _tool_name(getattr(call, "name", None))
            if not name:
                continue
            raw_id = getattr(call, "id", None)
            call_id = raw_id if isinstance(raw_id, str) and raw_id else uuid.uuid4().hex
            if isinstance(raw_id, str) and raw_id:
                if call_id in seen_tool_ids:
                    continue
                seen_tool_ids.add(call_id)
            raw_args = getattr(call, "args", None)
            args: ToolArgs = raw_args if isinstance(raw_args, dict) else {}
            state.pending_tools[call_id] = _PendingTool(name=name, started=time.monotonic())
            queue.put_nowait(ToolCallRequest(name=name, args=args, metadata={"call_id": call_id}))

    def _complete_pending_from_step(
        self, step: SDKStep, state: _AntigravitySessionState, status: str
    ) -> None:
        """Close any still-pending tool calls in a terminal ``TOOL_CALL`` step.

        Fallback for when the ``PostToolCallHook`` doesn't fire for a call. The
        hook pops a completed call from ``state.pending_tools``, so an id still
        present here wasn't completed by it — emit a :class:`ToolCallComplete`
        (without the payload, which only the hook carries). Popping here makes
        a later hook fire for the same id a no-op, so it completes once.

        :param step: The terminal ``TOOL_CALL`` step.
        :param state: The session state (its ``pending_tools`` is drained).
        :param status: The step's status name, e.g. ``"ERROR"`` / ``"DONE"``.
        """
        queue = state.active_queue
        if queue is None:
            return
        if status in ("ERROR", "TERMINAL_ERROR"):
            outcome = ToolCallStatus.ERROR
            error = getattr(step, "error", "") or None
        elif status == "CANCELED":
            outcome = ToolCallStatus.CANCELLED
            error = None
        else:
            outcome = ToolCallStatus.SUCCESS
            error = None
        for call in getattr(step, "tool_calls", None) or []:
            raw_id = getattr(call, "id", None)
            # id-less calls can't be matched to a pending entry; the hook is
            # their only completion path.
            if not (isinstance(raw_id, str) and raw_id):
                continue
            pending = state.pending_tools.pop(raw_id, None)
            if pending is None:
                continue  # already completed by the PostToolCallHook
            duration_ms = (time.monotonic() - pending.started) * 1000
            queue.put_nowait(
                ToolCallComplete(
                    name=pending.name,
                    status=outcome,
                    result=None,
                    error=error,
                    duration_ms=duration_ms,
                    metadata={"call_id": raw_id},
                )
            )

    @staticmethod
    def _cancelled_error_type() -> type[BaseException]:
        """Return the SDK's cancellation exception type, or a sentinel.

        Resolved lazily so the module imports without the SDK present. Falls
        back to a never-raised type when unavailable, making the ``except``
        clause a harmless no-match.

        :returns: ``google.antigravity.types.AntigravityCancelledError`` when
            resolvable, else :class:`_NeverRaisedError` (a no-match sentinel).
        """
        try:
            antigravity = _ensure_antigravity_sdk()
        except ImportError:
            return _NeverRaisedError
        err = getattr(antigravity.types, "AntigravityCancelledError", None)
        if isinstance(err, type) and issubclass(err, BaseException):
            return err
        return _NeverRaisedError

    async def _ensure_agent(
        self,
        session_key: str,
        *,
        model: str,
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> _AntigravitySessionState:
        """Return the session state with a current SDK agent + conversation.

        Rebuilds the agent when the model, system prompt, or tool set changes;
        otherwise reuses it to preserve conversation state. The state object
        is preserved across rebuilds so the agent's ``PostToolCallHook`` (which
        closes over it) stays valid.

        :param session_key: The Omnigent session id.
        :param model: The resolved model id to pin, e.g. ``"gemini-3.5-flash"``.
        :param system_prompt: The agent's system instructions.
        :param tools: Omnigent tool specs to expose to the agent.
        :returns: The session's :class:`_AntigravitySessionState`, with
            ``agent`` and ``conversation`` populated.
        """
        signature = (model, system_prompt, self._tool_signature(tools))
        state = self._session_states.get(session_key)
        if state is None:
            state = _AntigravitySessionState()
            self._session_states[session_key] = state

        if state.agent is not None and state.agent_signature == signature:
            return state

        if state.agent is not None:
            await self._close_agent(state.agent)
            state.agent = None
            state.conversation = None

        agent = await self._open_agent(
            state, model=model, system_prompt=system_prompt, tools=tools
        )
        state.agent = agent
        state.conversation = agent.conversation
        state.agent_signature = signature
        return state

    async def _open_agent(
        self,
        state: _AntigravitySessionState,
        *,
        model: str,
        system_prompt: str,
        tools: list[ToolSpec],
    ) -> SDKAgent:
        """Construct and open a ``google.antigravity.Agent``.

        Isolated SDK touchpoint. Builds a ``LocalAgentConfig`` from the
        resolved model / prompt / credentials / tools / hooks and enters the
        agent's async context. Omnigent's tools are exposed as callables
        (``LocalAgentConfig.tools``) routing through :attr:`_tool_executor`, so
        the agent runs them under policy. Two hooks are registered: a
        ``PreToolCallDecideHook`` that gates every tool against Omnigent's
        TOOL_CALL policy BEFORE it runs (see :meth:`_build_pre_tool_hook`), and
        a ``PostToolCallHook`` that surfaces a :class:`ToolCallComplete` for
        every tool the agent runs.

        :param state: The session state the tool-completion hook closes over.
        :param model: The resolved model id to pin.
        :param system_prompt: The agent's system instructions.
        :param tools: Omnigent tool specs to expose as callables.
        :returns: The opened SDK ``Agent``.
        """
        antigravity = _ensure_antigravity_sdk()
        config_kwargs: _StrAnyDict = {"system_instructions": system_prompt or None}
        sdk_tools = self._build_sdk_tools(tools)
        if sdk_tools:
            config_kwargs["tools"] = sdk_tools
        # Bind this agent's bridged-tool skip set into the hook below rather
        # than reading mutable executor state at fire time: the agent (and so
        # this hook) is rebuilt whenever the tool set changes, so the closure
        # always holds the right names and a concurrent turn on another session
        # can't corrupt them.
        bridged_tool_names = self._bridged_tool_name_set(tools)
        # The pre-tool decide hook runs first so a policy DENY blocks execution
        # before the post-tool hook (which only observes completed calls).
        config_kwargs["hooks"] = [
            self._build_pre_tool_hook(antigravity, bridged_tool_names),
            self._build_post_tool_hook(antigravity, state),
        ]
        config = self._build_local_agent_config(antigravity, model=model, kwargs=config_kwargs)
        agent = antigravity.Agent(config)
        # Agent is documented as an async context manager; enter it if so.
        if hasattr(agent, "__aenter__"):
            agent = await agent.__aenter__()
        return agent

    async def _evaluate_tool_call_policy(
        self, tool_name: str, tool_args: ToolArgs
    ) -> tuple[bool, str]:
        """Run a pre-execution TOOL_CALL policy evaluation for one tool call.

        The policy half of the SDK's pre-tool decide gate, mirroring
        :meth:`omnigent.inner.claude_sdk_executor.ClaudeSDKExecutor._evaluate_tool_call_policy`.
        Consults :attr:`_policy_evaluator` for the ``PHASE_TOOL_CALL`` phase and
        collapses the verdict to an allow/deny pair the SDK hook can return:

        - **ALLOW / no-match / no evaluator** -> ``(True, "")`` (proceed).
        - **DENY** -> ``(False, reason)`` (the SDK rejects the call so the tool
          never runs).
        - **ASK** -> route through :attr:`_elicitation_handler`; ``(True, "")``
          on approval, ``(False, reason)`` on decline or when no handler is
          wired (fail closed). A raw ASK normally only reaches here on a
          read-only evaluation path — the server usually collapses ASK to a
          hard ALLOW/DENY first.
        - **Unexpected verdict** -> ``(False, reason)`` (fail closed).

        :param tool_name: The tool's wire name, e.g. ``"run_command"`` (a
            bundled native tool) or ``"sys_shell"`` (a bridged Omnigent tool).
        :param tool_args: The tool call's argument dict.
        :returns: ``(allow, reason)`` — *reason* is the deny message surfaced
            to the model (empty when *allow* is ``True``).
        """
        policy_eval = getattr(self, "_policy_evaluator", None)
        if policy_eval is None:
            return True, ""
        verdict = await policy_eval(
            "PHASE_TOOL_CALL",
            {"name": tool_name, "arguments": tool_args},
        )
        action = getattr(verdict, "action", None)
        if action in ("POLICY_ACTION_ALLOW", "POLICY_ACTION_UNSPECIFIED", None):
            return True, ""
        reason = getattr(verdict, "reason", None)
        if action == "POLICY_ACTION_ASK":
            ask_reason = reason or "Approval required by Omnigent TOOL_CALL policy"
            if self._elicitation_handler is None:
                logger.warning(
                    "TOOL_CALL policy ASK had no elicitation handler; denying tool=%s reason=%s",
                    tool_name,
                    ask_reason,
                )
                return False, ask_reason
            logger.info(
                "TOOL_CALL policy requested approval tool=%s reason=%s", tool_name, ask_reason
            )
            if await self._elicitation_handler(tool_name, tool_args):
                return True, ""
            return False, ask_reason
        if action == "POLICY_ACTION_DENY":
            deny_reason = reason or "Denied by Omnigent TOOL_CALL policy"
            logger.info("TOOL_CALL policy denied tool=%s reason=%s", tool_name, deny_reason)
            return False, deny_reason
        fail_reason = f"Unexpected Omnigent TOOL_CALL policy verdict: {action!r}"
        logger.warning("TOOL_CALL policy failed closed tool=%s reason=%s", tool_name, fail_reason)
        return False, fail_reason

    def _build_pre_tool_hook(
        self, antigravity: ModuleType, bridged_tool_names: frozenset[str]
    ) -> SDKHook:
        """Build a ``PreToolCallDecideHook`` that enforces TOOL_CALL policy.

        The SDK invokes this BEFORE every tool runs with the pending
        ``ToolCall`` and honors the returned ``HookResult``: ``allow=False``
        rejects the call so the tool never executes — for BOTH the SDK's
        bundled native tools (``run_command``, file ops; gated in the SDK's
        ``_handle_tool_confirmation_request``) AND the bridged Omnigent tools
        (gated in the SDK's ``_handle_tool_call`` before it reaches the
        host-side runner). This is the executor's analogue of
        claude-sdk's ``can_use_tool`` gate.

        Double-evaluation guard: the bridged Omnigent tools (the ``tools`` the
        executor exposed for this turn) route through :attr:`_tool_executor` ->
        ``TurnContext.dispatch_tool`` -> the runner's ``ProxyMcpManager``, which
        ALREADY enforces TOOL_CALL + TOOL_RESULT policies server-side (see
        ``omnigent/runner/app.py`` "All tool calls go through AP:/mcp ... which
        enforces TOOL_CALL + TOOL_RESULT policies server-side"). Re-evaluating
        them here would double-count the same call (and could double-charge a
        cost-budget checkpoint), so the hook SKIPS bridged tools and only gates
        the tools the dispatch path never sees — chiefly the SDK's bundled
        native tools. This mirrors claude-sdk skipping its ``mcp__omnigent__*``
        prefix for the same reason.

        Scoping (the fix for the per-turn-state bypass): *bridged_tool_names* is
        bound into this hook's closure — NOT read from mutable executor state —
        and a fresh hook is built each time the agent is rebuilt (the tool set
        is part of the agent signature), so a concurrent or subsequent turn on
        another session can't swap the skip set out from under an in-flight
        call. The skip is also made precise so a name collision can't open a
        hole: a call to a bundled NATIVE tool is always gated even when its wire
        name equals a bridged tool's name. Native is detected at the call site
        (``ToolCall.name`` is a ``BuiltinTools`` enum member, or its wire name is
        a known native name — see :func:`_is_native_tool_call`), so only a
        genuinely bridged call is skipped.

        Defined inline because its base class only exists once the SDK is
        imported. Closes over the executor (``self``) so the gate reads the
        live :attr:`_policy_evaluator` / :attr:`_elicitation_handler`.

        :param antigravity: The imported ``google.antigravity`` module.
        :param bridged_tool_names: Wire names of the tools bridged through the
            dispatch path for this agent — the only names eligible to skip.
        :returns: A ``PreToolCallDecideHook`` for ``LocalAgentConfig.hooks``.
        """
        executor = self
        native_wire_names = _native_tool_wire_names(antigravity)

        class _OmnigentToolPolicyHook(antigravity.hooks.PreToolCallDecideHook):  # type: ignore[misc, name-defined]
            """Gates each SDK ``ToolCall`` against Omnigent's TOOL_CALL policy."""

            async def run(self, context: SDKHookContext, data: SDKToolCall) -> SDKHookResult:  # noqa: ARG002 — context unused; required by hook signature
                """Allow or deny one pending tool call before it runs.

                :param context: The SDK ``OperationContext`` (unused here).
                :param data: The SDK ``ToolCall`` about to execute.
                :returns: A ``HookResult`` — ``allow=False`` blocks the call.
                """
                name = _tool_name(getattr(data, "name", None))
                if not name:
                    return antigravity.types.HookResult(allow=True)
                # Skip ONLY a genuinely bridged call (already TOOL_CALL-gated
                # server-side on the dispatch path). A native tool is gated here
                # even if its wire name collides with a bridged name, so a
                # bridged-name collision can't smuggle a native call past policy.
                if name in bridged_tool_names and not _is_native_tool_call(
                    data, native_wire_names
                ):
                    return antigravity.types.HookResult(allow=True)
                raw_args = getattr(data, "args", None)
                args: ToolArgs = raw_args if isinstance(raw_args, dict) else {}
                allow, reason = await executor._evaluate_tool_call_policy(name, args)
                return antigravity.types.HookResult(allow=allow, message=reason)

        return _OmnigentToolPolicyHook()

    def _build_post_tool_hook(
        self, antigravity: ModuleType, state: _AntigravitySessionState
    ) -> SDKHook:
        """Build a ``PostToolCallHook`` that emits :class:`ToolCallComplete`.

        The SDK invokes this after every tool call with a ``ToolResult``; the
        hook pairs it to the originating :class:`ToolCallRequest` via
        ``state.pending_tools[id]`` (for duration) and enqueues a
        :class:`ToolCallComplete`. Defined inline because its base only exists
        once the SDK is imported.

        :param antigravity: The imported ``google.antigravity`` module.
        :param state: The session state (queue + pending-tool table) the hook
            reads at fire time.
        :returns: A ``PostToolCallHook`` instance for ``LocalAgentConfig.hooks``.
        """

        class _OmnigentToolCompleteHook(antigravity.hooks.PostToolCallHook):  # type: ignore[misc, name-defined]
            """Maps each SDK ``ToolResult`` onto a :class:`ToolCallComplete`."""

            async def run(self, context: SDKHookContext, data: SDKToolResult) -> None:  # noqa: ARG002 — context unused; required by hook signature
                """Enqueue a completion event for the finished tool call.

                :param context: The SDK ``HookContext`` (unused here).
                :param data: The SDK ``ToolResult`` for the finished call.
                """
                queue = state.active_queue
                if queue is None:
                    return
                raw_id = getattr(data, "id", None)
                call_id = raw_id if isinstance(raw_id, str) and raw_id else None
                pending = state.pending_tools.pop(call_id, None) if call_id else None
                # For an id'd call, a missing pending entry means the step-stream
                # fallback already completed it — skip to avoid a double emit.
                if call_id is not None and pending is None:
                    return
                name = _tool_name(getattr(data, "name", None)) or (pending.name if pending else "")
                duration_ms = (time.monotonic() - pending.started) * 1000 if pending else 0.0
                result = getattr(data, "result", None)
                error = getattr(data, "error", None)
                classification = classify_tool_result(result)
                status = ToolCallStatus.ERROR if error else classification.status
                message = error or (classification.error or None)
                queue.put_nowait(
                    ToolCallComplete(
                        name=name,
                        status=status,
                        result=result,
                        error=message,
                        duration_ms=duration_ms,
                        metadata={"call_id": call_id} if call_id else {},
                    )
                )

        return _OmnigentToolCompleteHook()

    def _bridged_tool_name_set(self, tools: list[ToolSpec]) -> frozenset[str]:
        """Return the names of the tools bridged through :attr:`_tool_executor`.

        These are the Omnigent tools exposed as SDK callables for the agent —
        exactly the names :meth:`_build_sdk_tools` would register. The result is
        bound into the pre-tool decide hook's closure (see
        :meth:`_build_pre_tool_hook`), which skips these names for genuinely
        bridged calls (they are TOOL_CALL-gated server-side on the dispatch
        path). Returns an empty set when no executor bridge is wired (the agent
        then runs native tools only, which the hook DOES gate).

        :param tools: Omnigent tool specs for the agent being built.
        :returns: The bridged tool names, or an empty set.
        """
        if not tools or self._tool_executor is None:
            return frozenset()
        return frozenset(
            name for tool in tools if isinstance((name := tool.get("name")), str) and name
        )

    def _build_sdk_tools(self, tools: list[ToolSpec]) -> list[SDKTool]:
        """Build SDK tools (plain callables) from Omnigent tool specs.

        The SDK introspects each callable's ``__name__`` / ``__doc__`` for its
        function declaration. Each routes through :attr:`_tool_executor`, so
        the agent reaches Omnigent's tool registry under policy. Returns ``[]``
        when there are no tools or no executor bridge yet (the agent then runs
        with its native + MCP tools only).

        :param tools: Omnigent tool specs (``name`` / ``description`` /
            ``parameters``).
        :returns: A list of named async callables, or ``[]``.
        """
        if not tools or self._tool_executor is None:
            return []
        sdk_tools: list[SDKTool] = []
        for tool in tools:
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = tool.get("description")
            description = description if isinstance(description, str) else ""
            sdk_tools.append(self._make_tool_callable(name, description))
        return sdk_tools

    def _make_tool_callable(self, tool_name: str, description: str) -> _ToolCallable:
        """Build a named async callable the SDK can register as a tool.

        Accepts the SDK's arg shape (kwargs, a single dict, or a JSON string)
        and forwards to :attr:`_tool_executor`. Its ``__name__`` / ``__doc__``
        are set so the SDK's function-declaration introspection picks up the
        name and description.

        :param tool_name: The Omnigent tool name, e.g. ``"sys_shell"``.
        :param description: Human-readable tool description for the model.
        :returns: An async callable bound to *tool_name*.
        """

        async def _invoke(*args: Any, **kwargs: Any) -> ToolResult:  # type: ignore[explicit-any]
            if self._tool_executor is None:
                return {"error": f"No tool executor for '{tool_name}'"}
            tool_args: _StrAnyDict = {}
            if kwargs:
                tool_args = dict(kwargs)
            elif args and isinstance(args[0], dict):
                tool_args = args[0]
            elif args and isinstance(args[0], str):
                try:
                    parsed = json.loads(args[0])
                    tool_args = parsed if isinstance(parsed, dict) else {"input": args[0]}
                except json.JSONDecodeError:
                    tool_args = {"input": args[0]}
            return await self._tool_executor(tool_name, tool_args)

        # The SDK builds the function declaration from these.
        _invoke.__name__ = tool_name
        _invoke.__qualname__ = tool_name
        _invoke.__doc__ = description or tool_name
        return _invoke

    def _build_local_agent_config(
        self,
        antigravity: ModuleType,
        *,
        model: str,
        kwargs: _StrAnyDict,
    ) -> SDKConfig:
        """Build a ``LocalAgentConfig``, passing only supported optional fields.

        Threads Gemini-native auth (``api_key`` or Vertex AI) and drops any
        field the installed SDK doesn't accept, rather than crashing on drift.

        :param antigravity: The imported ``google.antigravity`` module.
        :param model: The resolved model id to pin.
        :param kwargs: Base config kwargs (system instructions, tools, hooks).
        :returns: A ``LocalAgentConfig`` instance.
        """
        local_config_cls = antigravity.LocalAgentConfig
        supported = self._config_field_names(local_config_cls)
        candidate: _StrAnyDict = dict(kwargs)
        candidate["model"] = model
        if self._api_key:
            candidate["api_key"] = self._api_key
        if self._vertex:
            candidate["vertex"] = True
            if self._project:
                candidate["project"] = self._project
            if self._location:
                candidate["location"] = self._location
        filtered = {
            key: value for key, value in candidate.items() if supported is None or key in supported
        }
        return local_config_cls(**filtered)

    @staticmethod
    def _config_field_names(config_cls: Any) -> set[str] | None:  # type: ignore[explicit-any]
        """Best-effort set of accepted ``LocalAgentConfig`` field names.

        Inspects the constructor signature to drop unsupported kwargs. Returns
        ``None`` for a ``**kwargs`` constructor (signature not introspectable),
        in which case the caller passes every candidate through.

        :param config_cls: The SDK's ``LocalAgentConfig`` class.
        :returns: The set of accepted field names, or ``None`` when the
            signature can't be introspected.
        """
        import inspect

        try:
            params = inspect.signature(config_cls).parameters
        except (TypeError, ValueError):
            return None
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return None
        return {name for name in params if name != "self"}

    @staticmethod
    def _extract_usage(meta: SDKUsage) -> _StrAnyDict | None:
        """Map an SDK ``UsageMetadata`` to Omnigent's usage dict shape.

        :param meta: The most recent ``UsageMetadata`` from the turn's steps,
            or ``None`` when the SDK reported no usage.
        :returns: A usage dict with any of ``input_tokens`` / ``output_tokens``
            / ``total_tokens`` / ``cache_read_input_tokens``, or ``None`` when
            no usage was reported.
        """
        if meta is None:
            return None
        usage: _StrAnyDict = {}
        prompt_tokens = getattr(meta, "prompt_token_count", None)
        output_tokens = getattr(meta, "candidates_token_count", None)
        total_tokens = getattr(meta, "total_token_count", None)
        cached = getattr(meta, "cached_content_token_count", None)
        if prompt_tokens is not None:
            usage["input_tokens"] = prompt_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens
        if cached is not None:
            usage["cache_read_input_tokens"] = cached
        return usage or None

    @staticmethod
    async def _close_agent(agent: SDKAgent) -> None:
        """Best-effort close of an SDK agent's async context.

        :param agent: The SDK ``Agent`` to tear down.
        """
        closer = getattr(agent, "__aexit__", None)
        if closer is not None:
            try:
                await closer(None, None, None)
            except Exception:  # noqa: BLE001 — agent teardown is best-effort
                logger.debug("Antigravity agent close failed", exc_info=True)
            return
        aclose = getattr(agent, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — agent teardown is best-effort
                logger.debug("Antigravity agent aclose failed", exc_info=True)
