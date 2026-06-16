"""
Unit tests for :class:`omnigent.inner.antigravity_executor.AntigravityExecutor`.

The fakes here mirror the real ``google.antigravity`` streaming surface the
executor depends on: ``agent.conversation`` yields :class:`Step` objects from
``receive_steps()`` (text / reasoning deltas, tool calls, status, usage) as the
turn runs, a registered ``PreToolCallDecideHook`` gates each call before it
runs, a registered ``PostToolCallHook`` fires per tool completion with a
``ToolResult``, and ``conversation.cancel()`` aborts a running turn. They let
the streaming / tool-pairing / policy-gating / cancellation logic be tested
without the SDK package or network.

Policy tests additionally use a ``_FakePolicyEvaluator`` (scripted per-phase
verdicts) and a ``_FakeElicitationHandler`` matching the async callables the
harness ExecutorAdapter wires onto the executor in production.
"""

from __future__ import annotations

import asyncio
import collections
import enum
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from omnigent.inner import antigravity_executor as ag
from omnigent.inner.antigravity_executor import AntigravityExecutor, _latest_user_text
from omnigent.inner.executor import (
    ExecutorConfig,
    ExecutorError,
    ReasoningChunk,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    ToolCallStatus,
    TurnCancelled,
    TurnComplete,
)

# ── Fakes mirroring the real SDK streaming shapes ───────────────────────


class _StepType(enum.Enum):
    """Subset of ``google.antigravity.types.StepType`` the executor reads."""

    TEXT_RESPONSE = "TEXT_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    FINISH = "FINISH"


class _StepStatus(enum.Enum):
    """Subset of ``google.antigravity.types.StepStatus`` the executor reads."""

    ACTIVE = "ACTIVE"
    DONE = "DONE"
    CANCELED = "CANCELED"
    ERROR = "ERROR"
    TERMINAL_ERROR = "TERMINAL_ERROR"


class _StepSource(enum.Enum):
    """Subset of ``google.antigravity.types.StepSource``."""

    SYSTEM = "SYSTEM"
    USER = "USER"
    MODEL = "MODEL"


class _StepTarget(enum.Enum):
    """Subset of ``google.antigravity.types.StepTarget``."""

    USER = "USER"
    ENVIRONMENT = "ENVIRONMENT"


class _BuiltinTools(enum.Enum):
    """Subset of ``google.antigravity.types.BuiltinTools`` (native tool names).

    Members carry the SDK's wire name as their ``.value`` — the same shape the
    executor's native-vs-bridged discriminator reads. A ``ToolCall`` whose
    ``name`` is one of these members is a bundled native tool; bridged callables
    present a plain ``str`` name instead.
    """

    RUN_COMMAND = "run_command"
    VIEW_FILE = "view_file"
    EDIT_FILE = "edit_file"


class _AntigravityCancelledError(Exception):
    """Stand-in for ``google.antigravity.types.AntigravityCancelledError``."""


class _FakeToolCall:
    def __init__(self, name: str, args: dict[str, Any], call_id: str | None = None) -> None:
        self.name = name
        self.args = args
        self.id = call_id


class _FakeToolResult:
    def __init__(
        self,
        name: str,
        result: Any = None,
        error: str | None = None,
        call_id: str | None = None,
    ) -> None:
        self.name = name
        self.result = result
        self.error = error
        self.id = call_id
        self.exception = None


class _FakeHookResult:
    """Stand-in for ``google.antigravity.types.HookResult`` (allow/message)."""

    def __init__(self, *, allow: bool = True, message: str = "") -> None:
        self.allow = allow
        self.message = message


class _FakeSDKToolCall:
    """Stand-in for ``google.antigravity.types.ToolCall`` (pre-tool hook data)."""

    def __init__(self, name: str, args: dict[str, Any], call_id: str | None = None) -> None:
        self.name = name
        self.args = args
        self.id = call_id


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_token_count = 11
        self.candidates_token_count = 7
        self.total_token_count = 18
        self.cached_content_token_count = 2


class _FakeStep:
    """Mirror of ``google.antigravity.types.Step`` (the fields the executor reads)."""

    def __init__(
        self,
        *,
        step_type: _StepType | None = None,
        status: _StepStatus | None = None,
        content_delta: str = "",
        thinking_delta: str = "",
        tool_calls: list[_FakeToolCall] | None = None,
        error: str = "",
        usage_metadata: _FakeUsage | None = None,
        source: _StepSource = _StepSource.MODEL,
        target: _StepTarget = _StepTarget.USER,
    ) -> None:
        self.type = step_type
        self.status = status
        self.content_delta = content_delta
        self.thinking_delta = thinking_delta
        self.tool_calls = tool_calls or []
        self.error = error
        self.usage_metadata = usage_metadata
        # Default MODEL->USER (assistant-facing); set source=USER to model the
        # SDK echoing the user's own input back in the step stream.
        self.source = source
        self.target = target


@dataclass
class _YieldStep:
    """Turn-script action: ``receive_steps`` yields this step."""

    step: _FakeStep


@dataclass
class _FireToolResult:
    """Turn-script action: the SDK invokes each PostToolCallHook with this result."""

    tool_result: _FakeToolResult


@dataclass
class _RaiseCancelled:
    """Turn-script action: ``receive_steps`` raises the SDK's cancellation error."""


@dataclass
class _RaiseGeneric:
    """Turn-script action: ``receive_steps`` raises a generic (non-cancel) error."""

    message: str = "boom"


@dataclass
class _ExecToolCall:
    """Turn-script action: run a tool through the SDK's pre-tool decide gate.

    Mirrors the real ``_handle_tool_call`` flow: every registered
    ``PreToolCallDecideHook`` is consulted with the ``ToolCall`` first; only if
    they all allow does the tool "execute" and its ``PostToolCallHook`` fire.
    Denied calls are recorded (and skip execution) so a test can prove the
    policy gate blocked the tool BEFORE it ran.

    :param call: The SDK ``ToolCall`` presented to the gate.
    :param result: The ``ToolResult`` to surface via the post-tool hook when
        the call is allowed.
    """

    call: _FakeSDKToolCall
    result: _FakeToolResult


# A turn script is the ordered list of actions one ``receive_steps()`` replays.
_TurnAction = _YieldStep | _FireToolResult | _ExecToolCall | _RaiseCancelled | _RaiseGeneric


class _FakeConversation:
    """Mirror of ``google.antigravity.conversation.Conversation`` (read paths).

    Splits the registered hooks by base class so the pre-tool decide hook
    (consulted with a ``ToolCall``, returns a ``HookResult``) and the post-tool
    hook (fired with a ``ToolResult``) are driven on their correct surfaces.
    """

    def __init__(self, hooks: list[Any], scripts: collections.deque[list[_TurnAction]]) -> None:
        self._pre_hooks = [h for h in hooks if isinstance(h, _FakePreToolCallDecideHook)]
        self._post_hooks = [h for h in hooks if isinstance(h, _FakePostToolCallHook)]
        self._scripts = scripts
        self.sends: list[str] = []
        self.cancel_called = 0
        # Tools the gate allowed to run / denied, by name — lets policy tests
        # assert a denied tool never executed.
        self.executed_tools: list[str] = []
        self.denied_tools: list[str] = []

    async def send(self, prompt: Any, **_kwargs: Any) -> None:
        self.sends.append(prompt)

    async def _gate_allows(self, call: _FakeSDKToolCall) -> bool:
        """Run every pre-tool decide hook; deny if any returns ``allow=False``."""
        for hook in self._pre_hooks:
            res = await hook.run(SimpleNamespace(), call)
            if not res.allow:
                return False
        return True

    async def receive_steps(self) -> Any:
        script = self._scripts.popleft() if self._scripts else []
        for action in script:
            if isinstance(action, _YieldStep):
                yield action.step
            elif isinstance(action, _FireToolResult):
                # Direct PostToolCallHook fire (call already past the gate).
                for hook in self._post_hooks:
                    await hook.run(SimpleNamespace(), action.tool_result)
            elif isinstance(action, _ExecToolCall):
                # Full SDK flow: gate first, then execute + complete on allow.
                if await self._gate_allows(action.call):
                    self.executed_tools.append(action.call.name)
                    for hook in self._post_hooks:
                        await hook.run(SimpleNamespace(), action.result)
                else:
                    self.denied_tools.append(action.call.name)
            elif isinstance(action, _RaiseCancelled):
                raise _AntigravityCancelledError("cancelled")
            elif isinstance(action, _RaiseGeneric):
                raise RuntimeError(action.message)

    async def cancel(self) -> None:
        self.cancel_called += 1


class _FakeAgent:
    def __init__(self, config: Any, scripts: collections.deque[list[_TurnAction]]) -> None:
        self.config = config
        self._conversation = _FakeConversation(list(getattr(config, "hooks", []) or []), scripts)
        self.closed = False

    @property
    def conversation(self) -> _FakeConversation:
        return self._conversation

    async def __aenter__(self) -> _FakeAgent:
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True


class _FakeLocalAgentConfig:
    """Mirror of ``LocalAgentConfig`` — accepts exactly the fields the executor sets."""

    def __init__(
        self,
        *,
        system_instructions: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        vertex: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        tools: Any = None,
        hooks: Any = None,
    ) -> None:
        self.system_instructions = system_instructions
        self.model = model
        self.api_key = api_key
        self.vertex = vertex
        self.project = project
        self.location = location
        self.tools = tools
        self.hooks = hooks


class _FakePostToolCallHook:
    """Sub-classable stand-in for ``google.antigravity.hooks.PostToolCallHook``."""

    async def run(self, context: Any, data: Any) -> None:
        return None


class _FakePreToolCallDecideHook:
    """Sub-classable stand-in for ``hooks.PreToolCallDecideHook`` (returns HookResult)."""

    async def run(self, context: Any, data: Any) -> Any:
        return _FakeHookResult(allow=True)


class _FakePolicyVerdict:
    """Stand-in for the adapter's ``PolicyVerdictPayload`` (action + reason)."""

    def __init__(self, action: str, reason: str | None = None) -> None:
        self.action = action
        self.reason = reason


class _FakePolicyEvaluator:
    """Scripted policy evaluator matching the executor's ``_policy_evaluator``.

    Records every ``(phase, data)`` it is called with so tests can assert what
    the executor evaluated, and returns a per-phase verdict (defaulting to
    ALLOW for any phase without an explicit override).
    """

    def __init__(self, verdicts: dict[str, _FakePolicyVerdict] | None = None) -> None:
        self._verdicts = verdicts or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, phase: str, data: dict[str, Any]) -> _FakePolicyVerdict:
        self.calls.append((phase, data))
        return self._verdicts.get(phase, _FakePolicyVerdict("POLICY_ACTION_ALLOW"))


class _FakeElicitationHandler:
    """Stand-in elicitation handler returning a fixed approve/deny decision."""

    def __init__(self, approve: bool) -> None:
        self._approve = approve
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        self.calls.append((tool_name, tool_input))
        return self._approve


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scripts: list[list[_TurnAction]],
) -> dict[str, Any]:
    """Patch ``_ensure_antigravity_sdk`` to return a fake module.

    :param monkeypatch: pytest monkeypatch fixture.
    :param scripts: One turn-script (list of actions) per ``receive_steps`` call,
        consumed front-to-back across turns / agent rebuilds.
    :returns: A ``captured`` dict exposing the agents / configs built, so tests
        can assert on what the executor passed to the SDK.
    """
    queue: collections.deque[list[_TurnAction]] = collections.deque(scripts)
    captured: dict[str, Any] = {"agents": [], "configs": []}

    class _FakeHooks:
        PostToolCallHook = _FakePostToolCallHook
        PreToolCallDecideHook = _FakePreToolCallDecideHook

    class _FakeTypes:
        AntigravityCancelledError = _AntigravityCancelledError
        HookResult = _FakeHookResult
        ToolCall = _FakeSDKToolCall
        BuiltinTools = _BuiltinTools

    class _FakeModule:
        LocalAgentConfig = _FakeLocalAgentConfig
        hooks = _FakeHooks
        types = _FakeTypes

        @staticmethod
        def Agent(config: Any) -> _FakeAgent:
            agent = _FakeAgent(config, queue)
            captured["agents"].append(agent)
            captured["configs"].append(config)
            return agent

    monkeypatch.setattr(ag, "_ensure_antigravity_sdk", lambda: _FakeModule())
    return captured


async def _drain(
    executor: AntigravityExecutor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    config: ExecutorConfig | None = None,
    system_prompt: str = "sys",
) -> list[Any]:
    events: list[Any] = []
    async for event in executor.run_turn(
        messages, tools=tools or [], system_prompt=system_prompt, config=config
    ):
        events.append(event)
    return events


def _text_step(delta: str) -> _YieldStep:
    return _YieldStep(
        _FakeStep(
            step_type=_StepType.TEXT_RESPONSE, status=_StepStatus.ACTIVE, content_delta=delta
        )
    )


def _tool_call_step(call: _FakeToolCall, status: _StepStatus = _StepStatus.ACTIVE) -> _YieldStep:
    return _YieldStep(_FakeStep(step_type=_StepType.TOOL_CALL, status=status, tool_calls=[call]))


# ── Tests ───────────────────────────────────────────────────────────────


def test_latest_user_text_prefers_last_user_message() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    assert _latest_user_text(messages) == "second"


@pytest.mark.asyncio
async def test_streaming_maps_text_reasoning_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text/reasoning stream as separate deltas; usage + final text land on TurnComplete."""
    script: list[_TurnAction] = [
        _YieldStep(_FakeStep(status=_StepStatus.ACTIVE, thinking_delta="thinking...")),
        _text_step("Hello "),
        _text_step("world"),
        _YieldStep(
            _FakeStep(
                step_type=_StepType.FINISH, status=_StepStatus.DONE, usage_metadata=_FakeUsage()
            )
        ),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor(model="gemini-3-pro", api_key="k")

    events = await _drain(executor, [{"role": "user", "content": "hi", "session_id": "s1"}])

    # Two TextChunks prove deltas stream incrementally rather than as one blob —
    # if the executor reverted to a one-shot agent.chat() this would be 1 (or 0).
    texts = [e.text for e in events if isinstance(e, TextChunk)]
    assert texts == ["Hello ", "world"]
    reasoning = [e for e in events if isinstance(e, ReasoningChunk)]
    assert len(reasoning) == 1 and reasoning[0].delta == "thinking..."
    assert reasoning[0].event_type == "reasoning_text"

    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(completes) == 1
    # Final text is the accumulation of the streamed deltas.
    assert completes[0].response == "Hello world"
    # Usage maps the SDK's UsageMetadata field names onto Omnigent's keys.
    assert completes[0].usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "cache_read_input_tokens": 2,
    }


@pytest.mark.asyncio
async def test_user_echoed_step_not_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A USER-source step (the SDK echoing the prompt) must not leak into the output.

    Regression guard for a real bug a live turn surfaced: the SDK streams the
    user's own input back as a ``source=USER`` step; mapping its content_delta
    to a TextChunk put the prompt into the assistant's response.
    """
    script: list[_TurnAction] = [
        _YieldStep(
            _FakeStep(
                step_type=_StepType.TEXT_RESPONSE,
                status=_StepStatus.ACTIVE,
                content_delta="echoed user prompt",
                source=_StepSource.USER,
                target=_StepTarget.USER,
            )
        ),
        _text_step("the real reply"),  # MODEL->USER by default
        _YieldStep(_FakeStep(step_type=_StepType.FINISH, status=_StepStatus.DONE)),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    texts = [e.text for e in events if isinstance(e, TextChunk)]
    # Only the MODEL->USER reply — the USER-source echo is filtered out.
    assert texts == ["the real reply"]
    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert completes[0].response == "the real reply"


@pytest.mark.asyncio
async def test_tool_request_and_completion_paired(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool call yields a request, then the PostToolCallHook yields a paired completion."""
    script: list[_TurnAction] = [
        _tool_call_step(_FakeToolCall("sys_shell", {"cmd": "ls"}, call_id="t1")),
        _FireToolResult(_FakeToolResult("sys_shell", result={"ok": True}, call_id="t1")),
        _text_step("done"),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    requests = [e for e in events if isinstance(e, ToolCallRequest)]
    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    assert len(requests) == 1 and len(completes) == 1
    assert requests[0].name == "sys_shell"
    assert requests[0].args == {"cmd": "ls"}
    assert requests[0].metadata == {"call_id": "t1"}
    # Completion is paired to the request by call_id, carries the real result,
    # and is classified SUCCESS (no error on the ToolResult).
    assert completes[0].metadata == {"call_id": "t1"}
    assert completes[0].name == "sys_shell"
    assert completes[0].result == {"ok": True}
    assert completes[0].status == ToolCallStatus.SUCCESS
    # duration_ms is computed from the recorded request start; >= 0 proves the
    # pending-tool table was populated by the request and read by the hook.
    assert completes[0].duration_ms >= 0.0
    # Request precedes completion in the stream.
    assert events.index(requests[0]) < events.index(completes[0])


@pytest.mark.asyncio
async def test_tool_completion_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ToolResult carrying an error maps to a ToolCallComplete with ERROR status."""
    script: list[_TurnAction] = [
        _tool_call_step(_FakeToolCall("sys_shell", {}, call_id="t1")),
        _FireToolResult(_FakeToolResult("sys_shell", error="permission denied", call_id="t1")),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    assert len(completes) == 1
    # ERROR (not SUCCESS) because the ToolResult.error was set; the message is
    # surfaced so the transcript shows why the tool failed.
    assert completes[0].status == ToolCallStatus.ERROR
    assert completes[0].error == "permission denied"


@pytest.mark.asyncio
async def test_tool_result_payload_error_classified_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ToolResult whose *payload* carries an error (not ToolResult.error) → ERROR."""
    script: list[_TurnAction] = [
        _tool_call_step(_FakeToolCall("sys_shell", {}, call_id="t1")),
        # error=None, but the result payload self-describes as an error — this
        # exercises classify_tool_result's payload branch, not the .error path.
        _FireToolResult(_FakeToolResult("sys_shell", result={"error": "boom"}, call_id="t1")),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    assert len(completes) == 1
    assert completes[0].status == ToolCallStatus.ERROR


@pytest.mark.asyncio
async def test_tool_call_without_id_still_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An id-less tool call still emits one request and one (unpaired) completion."""
    script: list[_TurnAction] = [
        _tool_call_step(_FakeToolCall("sys_shell", {"cmd": "ls"}, call_id=None)),
        _FireToolResult(_FakeToolResult("sys_shell", result={"ok": True}, call_id=None)),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    requests = [e for e in events if isinstance(e, ToolCallRequest)]
    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    # Exactly one of each: the request gets a synthetic id (so it's still shown);
    # the id-less completion can't pair back, so its metadata is empty and it
    # falls back to the ToolResult's own name. The tool must still "close".
    assert len(requests) == 1
    assert len(completes) == 1
    assert completes[0].name == "sys_shell"
    assert completes[0].metadata == {}
    assert completes[0].duration_ms == 0.0


@pytest.mark.asyncio
async def test_tool_error_step_completes_without_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a TOOL_CALL step errors and the hook never fires, the step closes the tool."""
    call = _FakeToolCall("sys_shell", {"cmd": "ls"}, call_id="t1")
    script: list[_TurnAction] = [
        _tool_call_step(call, status=_StepStatus.ACTIVE),
        # No _FireToolResult: simulate the SDK surfacing the tool error outside
        # PostToolCallHook. The terminal TOOL_CALL ERROR step must still close it.
        _tool_call_step(call, status=_StepStatus.ERROR),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    # Without the step-stream fallback the tool would stay "open" (0 completions);
    # the fallback emits exactly one ERROR completion paired by call_id.
    assert len(completes) == 1
    assert completes[0].status == ToolCallStatus.ERROR
    assert completes[0].metadata == {"call_id": "t1"}
    # The turn itself is not failed — a tool error is not a turn-level error.
    assert any(isinstance(e, TurnComplete) for e in events)
    assert not any(isinstance(e, ExecutorError) for e in events)


@pytest.mark.asyncio
async def test_tool_completion_not_double_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both the hook and a terminal step fire, the tool completes exactly once."""
    call = _FakeToolCall("sys_shell", {"cmd": "ls"}, call_id="t1")
    script: list[_TurnAction] = [
        _tool_call_step(call, status=_StepStatus.ACTIVE),
        _FireToolResult(_FakeToolResult("sys_shell", result={"ok": True}, call_id="t1")),
        # A trailing DONE step for the same call — the fallback must see it as
        # already-completed (popped by the hook) and NOT emit a second event.
        _tool_call_step(call, status=_StepStatus.DONE),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    # 1, not 2: the hook completed it (with the real result) and popped the
    # pending entry, so the DONE-step fallback no-ops.
    assert len(completes) == 1
    assert completes[0].result == {"ok": True}


@pytest.mark.asyncio
async def test_tool_request_deduped_across_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same tool-call id appearing in multiple steps yields exactly one request."""
    call = _FakeToolCall("sys_shell", {"cmd": "ls"}, call_id="dup")
    script: list[_TurnAction] = [
        _tool_call_step(call, status=_StepStatus.ACTIVE),
        _tool_call_step(call, status=_StepStatus.DONE),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    requests = [e for e in events if isinstance(e, ToolCallRequest)]
    # 1, not 2: the SDK re-emits the same ToolCall across dispatch/execution
    # step transitions; the seen-id set must suppress the duplicate request.
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_terminal_error_step_yields_executor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TERMINAL_ERROR step surfaces an ExecutorError and suppresses TurnComplete."""
    script: list[_TurnAction] = [
        _text_step("partial"),
        _YieldStep(
            _FakeStep(
                step_type=_StepType.FINISH,
                status=_StepStatus.TERMINAL_ERROR,
                error="model exploded",
            )
        ),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    errors = [e for e in events if isinstance(e, ExecutorError)]
    assert len(errors) == 1
    assert errors[0].message == "model exploded"
    # TERMINAL_ERROR is non-retryable (a plain ERROR would be retryable).
    assert errors[0].retryable is False
    # No TurnComplete after a turn-level error — the workflow treats it as failed.
    assert not any(isinstance(e, TurnComplete) for e in events)


@pytest.mark.asyncio
async def test_error_step_without_message_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ERROR step with no error text still yields an ExecutorError (not a silent success)."""
    script: list[_TurnAction] = [
        _YieldStep(_FakeStep(step_type=_StepType.FINISH, status=_StepStatus.ERROR, error="")),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    errors = [e for e in events if isinstance(e, ExecutorError)]
    # An empty error string must not be reported as a successful (empty) turn;
    # the executor substitutes a generic message and a plain ERROR is retryable.
    assert len(errors) == 1
    assert errors[0].message
    assert errors[0].retryable is True
    assert not any(isinstance(e, TurnComplete) for e in events)


@pytest.mark.asyncio
async def test_empty_turn_yields_turn_complete_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn that streams no text ends as TurnComplete(response=None), not ''."""
    script: list[_TurnAction] = [
        _YieldStep(_FakeStep(step_type=_StepType.FINISH, status=_StepStatus.DONE))
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(completes) == 1
    # None (not "") is the load-bearing "produced nothing" signal documented on
    # TurnComplete; a regression to "" would change how the empty turn renders.
    assert completes[0].response is None
    assert not any(isinstance(e, TextChunk) for e in events)


@pytest.mark.asyncio
async def test_canceled_step_yields_turn_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CANCELED step surfaces TurnCancelled and no TurnComplete."""
    script: list[_TurnAction] = [
        _text_step("starting"),
        _YieldStep(_FakeStep(status=_StepStatus.CANCELED)),
    ]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    assert any(isinstance(e, TurnCancelled) for e in events)
    assert not any(isinstance(e, TurnComplete) for e in events)


@pytest.mark.asyncio
async def test_sdk_cancelled_error_yields_turn_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AntigravityCancelledError`` from the SDK maps to TurnCancelled, not ExecutorError."""
    script: list[_TurnAction] = [_text_step("starting"), _RaiseCancelled()]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    # The cancellation exception is caught specifically (via _cancelled_error_type
    # resolving the SDK's type) and reported as a clean cancel, not a failure.
    assert any(isinstance(e, TurnCancelled) for e in events)
    assert not any(isinstance(e, ExecutorError) for e in events)


@pytest.mark.asyncio
async def test_generic_turn_failure_yields_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-cancel exception from the SDK becomes a retryable ExecutorError."""
    script: list[_TurnAction] = [_text_step("partial"), _RaiseGeneric("kaboom")]
    _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    errors = [e for e in events if isinstance(e, ExecutorError)]
    assert len(errors) == 1
    assert "kaboom" in errors[0].message
    # retryable=True (unlike TERMINAL_ERROR) so the workflow picks RetryableLLMError;
    # also distinguishes a generic failure from a clean cancel (TurnCancelled).
    assert errors[0].retryable is True
    assert not any(isinstance(e, (TurnComplete, TurnCancelled)) for e in events)


@pytest.mark.asyncio
async def test_missing_sdk_yields_executor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> Any:
        raise ImportError("no google-antigravity")

    monkeypatch.setattr(ag, "_ensure_antigravity_sdk", _raise)
    executor = AntigravityExecutor()

    events = await _drain(executor, [{"role": "user", "content": "q"}])

    assert len(events) == 1
    assert isinstance(events[0], ExecutorError)
    assert "google-antigravity" in events[0].message


@pytest.mark.asyncio
async def test_sys_tools_exposed_as_callables_routing_through_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omnigent tools become callable SDK tools whose calls hit ``_tool_executor``.

    This is what lets an Antigravity agent drive Omnigent's sys / sub-agent
    tools under policy (needed to run Polly / Debby).
    """
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("done")]])
    executor = AntigravityExecutor()

    calls: list[dict[str, Any]] = []

    async def _fake_tool_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append({"name": name, "args": args})
        return {"ok": True}

    # The harness ExecutorAdapter assigns this in production; set it directly.
    executor._tool_executor = _fake_tool_executor

    tool_specs = [
        {
            "name": "sys_shell",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ]

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}], tool_specs)

    sdk_tools = captured["configs"][0].tools
    assert sdk_tools is not None and len(sdk_tools) == 1
    sdk_tool = sdk_tools[0]
    # LocalAgentConfig.tools is list[Callable]; the SDK reads __name__/__doc__.
    assert callable(sdk_tool)
    assert sdk_tool.__name__ == "sys_shell"
    assert sdk_tool.__doc__ == "Run a shell command"

    # Invoking the callable (kwargs form) routes back through the bridge.
    assert await sdk_tool(cmd="ls") == {"ok": True}
    # Single-dict argument form also works (SDK arg-shape tolerance).
    assert await sdk_tool({"cmd": "pwd"}) == {"ok": True}
    assert calls == [
        {"name": "sys_shell", "args": {"cmd": "ls"}},
        {"name": "sys_shell", "args": {"cmd": "pwd"}},
    ]


@pytest.mark.asyncio
async def test_no_tool_executor_means_no_sdk_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a tool-executor bridge, no SDK tools are built (agent uses native)."""
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("done")]])
    executor = AntigravityExecutor()  # _tool_executor stays None

    await _drain(
        executor,
        [{"role": "user", "content": "go"}],
        [{"name": "sys_shell", "description": "", "parameters": {}}],
    )

    assert captured["configs"][0].tools is None


@pytest.mark.asyncio
async def test_agent_reused_across_turns_same_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second turn on the same session reuses the cached agent + conversation."""
    captured = _install_fake_sdk(
        monkeypatch, scripts=[[_text_step("one-reply")], [_text_step("two-reply")]]
    )
    executor = AntigravityExecutor()

    await _drain(executor, [{"role": "user", "content": "one", "session_id": "s1"}])
    await _drain(executor, [{"role": "user", "content": "two", "session_id": "s1"}])

    # Exactly one agent built across two turns — the signature was unchanged so
    # the cached agent (and its SDK conversation state) was reused.
    assert len(captured["agents"]) == 1
    assert captured["agents"][0].conversation.sends == ["one", "two"]


@pytest.mark.asyncio
async def test_model_switch_rebuilds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-turn model override changes the signature and rebuilds the agent."""
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("a")], [_text_step("b")]])
    executor = AntigravityExecutor(model="gemini-3-pro")

    await _drain(executor, [{"role": "user", "content": "one", "session_id": "s1"}])
    await _drain(
        executor,
        [{"role": "user", "content": "two", "session_id": "s1"}],
        config=ExecutorConfig(model="gemini-3-flash"),
    )

    # Two agents: the model changed (gemini-3-pro -> gemini-3-flash), which is
    # part of the agent signature, so the executor rebuilt rather than reused.
    assert len(captured["agents"]) == 2
    assert captured["configs"][0].model == "gemini-3-pro"
    assert captured["configs"][1].model == "gemini-3-flash"


@pytest.mark.asyncio
async def test_default_model_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no model on the executor or per-turn config, the built-in default is pinned."""
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("ok")]])
    executor = AntigravityExecutor()  # no model anywhere

    await _drain(executor, [{"role": "user", "content": "hi", "session_id": "s1"}])

    # Pins _ANTIGRAVITY_DEFAULT_MODEL; changing the default must update this.
    assert captured["configs"][0].model == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_system_prompt_change_rebuilds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed system_prompt is part of the agent signature, forcing a rebuild."""
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("a")], [_text_step("b")]])
    executor = AntigravityExecutor()

    await _drain(
        executor, [{"role": "user", "content": "one", "session_id": "s1"}], system_prompt="first"
    )
    await _drain(
        executor, [{"role": "user", "content": "two", "session_id": "s1"}], system_prompt="second"
    )

    # Two agents: system_prompt is in the (model, system_prompt, tools) signature,
    # so changing it rebuilds. A regression dropping system_prompt would be 1.
    assert len(captured["agents"]) == 2


@pytest.mark.asyncio
async def test_api_key_and_vertex_threaded_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_key and Vertex (project/location) reach LocalAgentConfig; base_url never does."""
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("ok")], [_text_step("ok")]])

    key_exec = AntigravityExecutor(api_key="gem-key")
    await _drain(key_exec, [{"role": "user", "content": "hi", "session_id": "s1"}])
    cfg = captured["configs"][0]
    assert cfg.api_key == "gem-key"
    # Vertex left unset on the API-key path.
    assert cfg.vertex is None

    vertex_exec = AntigravityExecutor(vertex=True, project="my-proj", location="us-central1")
    await _drain(vertex_exec, [{"role": "user", "content": "hi", "session_id": "s2"}])
    vcfg = captured["configs"][1]
    assert vcfg.vertex is True
    assert vcfg.project == "my-proj"
    assert vcfg.location == "us-central1"
    # The SDK config has no base_url field — the executor must never set one.
    assert not hasattr(vcfg, "base_url")


@pytest.mark.asyncio
async def test_close_session_closes_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """close_session() tears down the cached SDK agent for that session."""
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("ok")]])
    executor = AntigravityExecutor()

    await _drain(executor, [{"role": "user", "content": "hi", "session_id": "s1"}])
    agent = captured["agents"][0]
    assert agent.closed is False  # still open after the turn

    await executor.close_session("s1")
    # _close_agent awaited the agent's __aexit__, releasing the SDK connection.
    assert agent.closed is True


# ── Interrupt (cancellation) tests — real deterministic sync gates ──────


class _BlockingConversation:
    """Conversation that streams one delta, then blocks until cancel() releases it.

    :param raise_on_release: when True, ``receive_steps`` raises the SDK
        cancellation error after the gate opens (the "SDK reports a cancel"
        path); when False it simply ends the stream cleanly (the "cancel ended
        the turn quietly" path that exercises the ``interrupt_requested`` gate).
    """

    def __init__(self, gate: asyncio.Event, raise_on_release: bool) -> None:
        self._gate = gate
        self._raise_on_release = raise_on_release
        self.sends: list[str] = []
        self.cancel_called = 0

    async def send(self, prompt: Any, **_kw: Any) -> None:
        self.sends.append(prompt)

    async def receive_steps(self) -> Any:
        yield _FakeStep(
            step_type=_StepType.TEXT_RESPONSE, status=_StepStatus.ACTIVE, content_delta="streaming"
        )
        await self._gate.wait()  # blocked until cancel() releases us
        if self._raise_on_release:
            raise _AntigravityCancelledError("cancelled")

    async def cancel(self) -> None:
        self.cancel_called += 1
        self._gate.set()


def _install_blocking_sdk(
    monkeypatch: pytest.MonkeyPatch, gate: asyncio.Event, *, raise_on_release: bool
) -> dict[str, Any]:
    """Install a fake SDK whose conversation blocks mid-turn until cancelled."""
    captured: dict[str, Any] = {}

    class _BlockingAgent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._conversation = _BlockingConversation(gate, raise_on_release)
            captured["conversation"] = self._conversation

        @property
        def conversation(self) -> _BlockingConversation:
            return self._conversation

        async def __aenter__(self) -> _BlockingAgent:
            return self

        async def __aexit__(self, *_a: object) -> None:
            return None

    class _FakeHooks:
        PostToolCallHook = _FakePostToolCallHook
        PreToolCallDecideHook = _FakePreToolCallDecideHook

    class _FakeTypes:
        AntigravityCancelledError = _AntigravityCancelledError
        HookResult = _FakeHookResult
        ToolCall = _FakeSDKToolCall
        BuiltinTools = _BuiltinTools

    class _FakeModule:
        LocalAgentConfig = _FakeLocalAgentConfig
        hooks = _FakeHooks
        types = _FakeTypes

        @staticmethod
        def Agent(config: Any) -> _BlockingAgent:
            return _BlockingAgent(config)

    monkeypatch.setattr(ag, "_ensure_antigravity_sdk", lambda: _FakeModule())
    return captured


async def _drive_until_first_text(
    executor: AntigravityExecutor, collected: list[Any], first_text: asyncio.Event
) -> None:
    async for event in executor.run_turn(
        [{"role": "user", "content": "go", "session_id": "s1"}], tools=[], system_prompt="sys"
    ):
        collected.append(event)
        if isinstance(event, TextChunk):
            first_text.set()


@pytest.mark.parametrize("raise_on_release", [True, False])
@pytest.mark.asyncio
async def test_interrupt_session_cancels_running_turn(
    monkeypatch: pytest.MonkeyPatch, raise_on_release: bool
) -> None:
    """interrupt_session cancels an in-flight turn -> TurnCancelled, no TurnComplete.

    Deterministic race: the conversation blocks inside ``receive_steps`` after
    streaming one delta; we interrupt only after observing that delta (so the
    turn is provably mid-flight). The two parametrized cases cover both ways the
    SDK can react to ``cancel()``: raising ``AntigravityCancelledError``
    (raise_on_release=True), or ending the stream cleanly so the
    ``interrupt_requested`` gate in run_turn must convert it to TurnCancelled
    (raise_on_release=False).
    """
    gate = asyncio.Event()
    first_text = asyncio.Event()
    captured = _install_blocking_sdk(monkeypatch, gate, raise_on_release=raise_on_release)
    executor = AntigravityExecutor()

    collected: list[Any] = []
    task = asyncio.create_task(_drive_until_first_text(executor, collected, first_text))
    # Wait until the turn has streamed its first delta (provably mid-flight)
    # before interrupting — this is the deterministic race window.
    await asyncio.wait_for(first_text.wait(), timeout=5)

    interrupted = await executor.interrupt_session("s1")
    # Assert the cancel landed BEFORE awaiting the task: a broken interrupt that
    # skips conversation.cancel() leaves the producer parked on the gate forever,
    # so checking here fails crisply ("cancel never called") instead of as an
    # opaque 5s task timeout below.
    assert interrupted is True  # a live conversation was found and asked to cancel
    assert captured["conversation"].cancel_called == 1  # cancel reached the SDK boundary

    await asyncio.wait_for(task, timeout=5)

    # Either path must surface a clean cancel and never a TurnComplete.
    assert any(isinstance(e, TurnCancelled) for e in collected)
    assert not any(isinstance(e, TurnComplete) for e in collected)


@pytest.mark.asyncio
async def test_interrupt_session_unknown_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """interrupt_session on a session with no open conversation returns False."""
    _install_fake_sdk(monkeypatch, scripts=[])
    executor = AntigravityExecutor()
    assert await executor.interrupt_session("never-started") is False


# ── Policy enforcement tests (TOOL_CALL / LLM_REQUEST / LLM_RESPONSE) ────


def _exec_tool(name: str, args: dict[str, Any], call_id: str = "t1") -> _ExecToolCall:
    """Build an _ExecToolCall that gates ``name`` then completes it on allow."""
    return _ExecToolCall(
        call=_FakeSDKToolCall(name, args, call_id=call_id),
        result=_FakeToolResult(name, result={"ok": True}, call_id=call_id),
    )


async def _async_ok() -> dict[str, Any]:
    """Minimal bridged-tool executor result used by the skip test."""
    return {"ok": True}


@pytest.mark.asyncio
async def test_tool_call_policy_deny_blocks_native_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TOOL_CALL-phase DENY prevents a native tool from executing.

    The pre-tool decide hook must consult the policy evaluator and return
    ``allow=False`` so the SDK rejects the call BEFORE running it. Without the
    hook (the bug) the tool would execute and the evaluator would never be
    asked for PHASE_TOOL_CALL.
    """
    deny = _FakePolicyVerdict("POLICY_ACTION_DENY", reason="run_command blocked by operator")
    evaluator = _FakePolicyEvaluator({"PHASE_TOOL_CALL": deny})
    # ``run_command`` is a bundled NATIVE tool (not bridged), so it is not in
    # the per-turn bridged set and the hook must gate it.
    script: list[_TurnAction] = [
        _exec_tool("run_command", {"command": "rm -rf /"}),
        _text_step("done"),
    ]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    conversation = captured["agents"][0].conversation
    # The gate denied the call: it was recorded as denied and NEVER executed.
    assert conversation.denied_tools == ["run_command"]
    assert conversation.executed_tools == []
    # The evaluator was consulted for the TOOL_CALL phase with the call args.
    tool_phase_calls = [d for (p, d) in evaluator.calls if p == "PHASE_TOOL_CALL"]
    assert tool_phase_calls == [{"name": "run_command", "arguments": {"command": "rm -rf /"}}]


@pytest.mark.asyncio
async def test_tool_call_policy_allow_runs_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ALLOW verdict lets the native tool execute (gate is not over-broad)."""
    evaluator = _FakePolicyEvaluator()  # defaults every phase to ALLOW
    script: list[_TurnAction] = [_exec_tool("run_command", {"command": "ls"}), _text_step("done")]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    conversation = captured["agents"][0].conversation
    assert conversation.executed_tools == ["run_command"]
    assert conversation.denied_tools == []


@pytest.mark.asyncio
async def test_tool_call_policy_skips_bridged_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bridged Omnigent tools are NOT re-evaluated here (gated server-side).

    They route through the dispatch path which already enforces TOOL_CALL
    policy; double-evaluating would double-count (and could double-charge a
    cost budget). The hook must let them through without calling the evaluator.
    """
    # A DENY verdict would block the tool IF the hook evaluated it — proving the
    # skip means asserting the tool still runs despite the deny.
    deny = _FakePolicyVerdict("POLICY_ACTION_DENY", reason="should not be consulted")
    evaluator = _FakePolicyEvaluator({"PHASE_TOOL_CALL": deny})
    script: list[_TurnAction] = [_exec_tool("sys_shell", {"cmd": "ls"}), _text_step("done")]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator
    # Wire a tool executor + expose ``sys_shell`` so it is a BRIDGED tool.
    executor._tool_executor = lambda name, args: _async_ok()
    tool_specs = [{"name": "sys_shell", "description": "shell", "parameters": {}}]

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}], tool_specs)

    conversation = captured["agents"][0].conversation
    # Ran despite the DENY: the hook skipped the bridged tool entirely.
    assert conversation.executed_tools == ["sys_shell"]
    assert conversation.denied_tools == []
    # The evaluator was never asked about the TOOL_CALL phase for this tool.
    assert not any(p == "PHASE_TOOL_CALL" for (p, _d) in evaluator.calls)


@pytest.mark.asyncio
async def test_tool_call_policy_ask_approved_runs_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TOOL_CALL ASK routes through the elicitation handler; approve -> run."""
    ask = _FakePolicyVerdict("POLICY_ACTION_ASK", reason="approve this command?")
    evaluator = _FakePolicyEvaluator({"PHASE_TOOL_CALL": ask})
    handler = _FakeElicitationHandler(approve=True)
    script: list[_TurnAction] = [_exec_tool("run_command", {"command": "ls"}), _text_step("done")]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator
    executor._elicitation_handler = handler

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    conversation = captured["agents"][0].conversation
    assert conversation.executed_tools == ["run_command"]
    # The elicitation handler was consulted with the tool name + args.
    assert handler.calls == [("run_command", {"command": "ls"})]


@pytest.mark.asyncio
async def test_tool_call_policy_ask_declined_blocks_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TOOL_CALL ASK declined by the elicitation handler blocks the tool."""
    ask = _FakePolicyVerdict("POLICY_ACTION_ASK")
    evaluator = _FakePolicyEvaluator({"PHASE_TOOL_CALL": ask})
    handler = _FakeElicitationHandler(approve=False)
    script: list[_TurnAction] = [_exec_tool("run_command", {"command": "ls"}), _text_step("done")]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator
    executor._elicitation_handler = handler

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    conversation = captured["agents"][0].conversation
    assert conversation.denied_tools == ["run_command"]
    assert conversation.executed_tools == []


@pytest.mark.asyncio
async def test_tool_call_policy_ask_without_handler_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TOOL_CALL ASK with no elicitation handler denies (fail closed)."""
    ask = _FakePolicyVerdict("POLICY_ACTION_ASK")
    evaluator = _FakePolicyEvaluator({"PHASE_TOOL_CALL": ask})
    script: list[_TurnAction] = [_exec_tool("run_command", {"command": "ls"}), _text_step("done")]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator  # no _elicitation_handler wired

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    conversation = captured["agents"][0].conversation
    assert conversation.denied_tools == ["run_command"]
    assert conversation.executed_tools == []


@pytest.mark.asyncio
async def test_no_policy_evaluator_does_not_gate_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no policy evaluator wired, the pre-tool hook is a no-op (tool runs)."""
    script: list[_TurnAction] = [_exec_tool("run_command", {"command": "ls"}), _text_step("done")]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()  # _policy_evaluator stays None

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    conversation = captured["agents"][0].conversation
    assert conversation.executed_tools == ["run_command"]
    assert conversation.denied_tools == []


@pytest.mark.asyncio
async def test_llm_request_policy_deny_aborts_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM_REQUEST DENY aborts the turn with ExecutorError and no model call.

    The gate runs before the producer is spawned, so ``conversation.send`` is
    never reached and no agent turn runs. Without the gate (the bug) the turn
    would proceed and stream a reply.
    """
    deny = _FakePolicyVerdict("POLICY_ACTION_DENY", reason="prompt contains a banned phrase")
    evaluator = _FakePolicyEvaluator({"PHASE_LLM_REQUEST": deny})
    # A normal turn script — it must NOT run because the request is denied first.
    captured = _install_fake_sdk(monkeypatch, scripts=[[_text_step("should not stream")]])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    errors = [e for e in events if isinstance(e, ExecutorError)]
    assert len(errors) == 1
    assert "denied by policy" in errors[0].message
    assert "prompt contains a banned phrase" in errors[0].message
    # No model call: the producer never sent the prompt, and no reply streamed.
    assert not any(isinstance(e, (TextChunk, TurnComplete)) for e in events)
    # The agent was built lazily AFTER the request gate, so no agent exists.
    assert captured["agents"] == []
    # The evaluator saw the request phase; it never reached the response phase.
    phases = [p for (p, _d) in evaluator.calls]
    assert phases == ["PHASE_LLM_REQUEST"]


@pytest.mark.asyncio
async def test_llm_request_policy_allow_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ALLOW request verdict lets the turn run normally (gate not over-broad)."""
    evaluator = _FakePolicyEvaluator()  # ALLOW for every phase
    _install_fake_sdk(monkeypatch, scripts=[[_text_step("hello")]])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    assert [e.text for e in events if isinstance(e, TextChunk)] == ["hello"]
    assert any(isinstance(e, TurnComplete) for e in events)


@pytest.mark.asyncio
async def test_llm_response_policy_deny_blocks_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM_RESPONSE DENY blocks the response: ExecutorError, no TurnComplete.

    Evaluated after the stream completes but before TurnComplete, so the
    generated text is never emitted as a completed turn. Without the gate the
    turn would complete and the (policy-violating) response would be persisted.
    """
    deny = _FakePolicyVerdict("POLICY_ACTION_DENY", reason="response leaked a secret")
    evaluator = _FakePolicyEvaluator({"PHASE_LLM_RESPONSE": deny})
    _install_fake_sdk(monkeypatch, scripts=[[_text_step("the secret is hunter2")]])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    errors = [e for e in events if isinstance(e, ExecutorError)]
    assert len(errors) == 1
    assert "response denied by policy" in errors[0].message
    assert "response leaked a secret" in errors[0].message
    # The text streamed live (deltas can't be un-sent), but the turn must NOT
    # complete — the DENY replaces TurnComplete with the ExecutorError.
    assert not any(isinstance(e, TurnComplete) for e in events)
    # Both phases were evaluated, response last, carrying the generated text.
    phases = [p for (p, _d) in evaluator.calls]
    assert phases == ["PHASE_LLM_REQUEST", "PHASE_LLM_RESPONSE"]
    resp_data = next(d for (p, d) in evaluator.calls if p == "PHASE_LLM_RESPONSE")
    assert resp_data["text_preview"] == "the secret is hunter2"


@pytest.mark.asyncio
async def test_llm_response_deny_matches_peer_stream_then_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_RESPONSE is a stream-then-verdict turn-completion gate, matching the peers.

    Documents (and pins) the deliberate cross-executor behaviour for FINDING 2:
    Antigravity does NOT buffer deltas until the LLM_RESPONSE verdict — it
    streams ``TextChunk``\\ s live and evaluates PHASE_LLM_RESPONSE only after the
    stream ends, exactly like ``ClaudeSDKExecutor`` (which yields ``TextChunk``
    per content-block delta and gates the response phase afterwards) and
    ``OpenAIAgentsSDKExecutor`` (which streams ``TextChunk`` and does not gate the
    response phase at all). So a DENY blocks TurnComplete / persistence but the
    delta the client already saw cannot be un-sent. Buffering would diverge from
    both peers unilaterally, so it is intentionally not done here.

    This test asserts the precise observable contract: the delta IS emitted
    (proving no buffering), it is emitted BEFORE the response verdict runs, and
    the verdict still blocks completion.
    """
    deny = _FakePolicyVerdict("POLICY_ACTION_DENY", reason="blocked output")
    evaluator = _FakePolicyEvaluator({"PHASE_LLM_RESPONSE": deny})
    _install_fake_sdk(monkeypatch, scripts=[[_text_step("streamed-delta")]])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator

    events = await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}])

    # The delta WAS streamed live (not buffered/withheld pending the verdict).
    texts = [e.text for e in events if isinstance(e, TextChunk)]
    assert texts == ["streamed-delta"]
    # The response verdict ran AFTER the stream produced its text (the recorded
    # text_preview proves the evaluator saw the already-streamed content).
    resp_data = next(d for (p, d) in evaluator.calls if p == "PHASE_LLM_RESPONSE")
    assert resp_data["text_preview"] == "streamed-delta"
    # The DENY still blocks turn completion: ExecutorError replaces TurnComplete.
    assert any(isinstance(e, ExecutorError) for e in events)
    assert not any(isinstance(e, TurnComplete) for e in events)


# ── FINDING 1 regression tests: collision-safe + per-agent skip set ──────


@pytest.mark.asyncio
async def test_native_tool_with_bridged_name_collision_still_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native tool is policy-gated even when its name collides with a bridged tool.

    Regression for the bypass: the old hook skipped any call whose bare name was
    in the bridged set, so a bundled NATIVE tool (e.g. ``view_file``) sharing a
    name with a bridged Omnigent tool would skip TOOL_CALL policy entirely. The
    fix gates a native call regardless of the name collision; it must recognise
    native via BOTH the wire-name set (string-named call) and the
    ``BuiltinTools`` enum (enum-named call).
    """
    deny = _FakePolicyVerdict("POLICY_ACTION_DENY", reason="view_file blocked by operator")
    evaluator = _FakePolicyEvaluator({"PHASE_TOOL_CALL": deny})
    # Two native ``view_file`` calls that collide with the bridged name — one
    # presented as a plain string name, one as a ``BuiltinTools`` enum member —
    # plus a genuinely bridged ``sys_shell`` call that must still be skipped.
    script: list[_TurnAction] = [
        _ExecToolCall(
            call=_FakeSDKToolCall("view_file", {"path": "/etc/passwd"}, call_id="n1"),
            result=_FakeToolResult("view_file", result={"ok": True}, call_id="n1"),
        ),
        _ExecToolCall(
            call=_FakeSDKToolCall(_BuiltinTools.VIEW_FILE, {"path": "/secrets"}, call_id="n2"),
            result=_FakeToolResult("view_file", result={"ok": True}, call_id="n2"),
        ),
        _exec_tool("sys_shell", {"cmd": "ls"}, call_id="b1"),
        _text_step("done"),
    ]
    captured = _install_fake_sdk(monkeypatch, scripts=[script])
    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator
    executor._tool_executor = lambda name, args: _async_ok()
    # Expose BOTH names as bridged — ``view_file`` deliberately collides with the
    # native tool, ``sys_shell`` does not.
    tool_specs = [
        {"name": "view_file", "description": "bridged view", "parameters": {}},
        {"name": "sys_shell", "description": "shell", "parameters": {}},
    ]

    await _drain(executor, [{"role": "user", "content": "go", "session_id": "s1"}], tool_specs)

    conversation = captured["agents"][0].conversation
    # Both native ``view_file`` calls were gated and DENIED despite the bridged
    # name collision; only the non-colliding bridged ``sys_shell`` ran. The
    # second call's recorded name is the ``BuiltinTools`` enum member (the SDK's
    # native-call shape), so resolve to wire names before comparing.
    denied_wire = [getattr(n, "value", n) for n in conversation.denied_tools]
    assert denied_wire == ["view_file", "view_file"]
    assert conversation.executed_tools == ["sys_shell"]
    # Policy was consulted for each native call (and only those) — the bridged
    # ``sys_shell`` was skipped, so it never reached the evaluator. The hook
    # passes the resolved wire name to the evaluator for both call shapes.
    tool_phase_names = [d["name"] for (p, d) in evaluator.calls if p == "PHASE_TOOL_CALL"]
    assert tool_phase_names == ["view_file", "view_file"]


class _GatingBlockingConversation:
    """Conversation that blocks mid-turn, then runs a tool through the pre-gate.

    Models a turn that is provably in flight when a *second*, concurrent turn
    runs to completion: it streams one delta, blocks on ``gate`` (so the test
    can interleave another session's full turn), and only after release drives
    its scripted ``ToolCall`` through every registered ``PreToolCallDecideHook``
    — exactly as the real ``_handle_tool_call`` flow does. Whether the call ran
    or was denied is recorded so the test can prove which bridged skip set the
    hook consulted at fire time.

    :param hooks: The agent's registered hooks (the pre-tool decide hooks are
        consulted; others are ignored).
    :param gate: Opened by the test once the concurrent turn has finished, so
        this turn's tool-gate fires strictly afterwards.
    :param call: The ``ToolCall`` to push through the gate after release.
    """

    def __init__(self, hooks: list[Any], gate: asyncio.Event, call: _FakeSDKToolCall) -> None:
        self._pre_hooks = [h for h in hooks if isinstance(h, _FakePreToolCallDecideHook)]
        self._gate = gate
        self._call = call
        self.sends: list[str] = []
        self.cancel_called = 0
        self.executed_tools: list[str] = []
        self.denied_tools: list[str] = []

    async def send(self, prompt: Any, **_kw: Any) -> None:
        self.sends.append(prompt)

    async def receive_steps(self) -> Any:
        yield _FakeStep(
            step_type=_StepType.TEXT_RESPONSE, status=_StepStatus.ACTIVE, content_delta="a-stream"
        )
        await self._gate.wait()  # held open until the concurrent turn completes
        allow = True
        for hook in self._pre_hooks:
            res = await hook.run(SimpleNamespace(), self._call)
            if not res.allow:
                allow = False
                break
        if allow:
            self.executed_tools.append(self._call.name)
        else:
            self.denied_tools.append(self._call.name)

    async def cancel(self) -> None:
        self.cancel_called += 1


@pytest.mark.asyncio
async def test_bridged_skip_set_not_corrupted_by_concurrent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent turn on another session can't corrupt an in-flight hook's skip set.

    Regression for the per-turn-state bypass: the skip set used to live on the
    executor instance (``self._bridged_tool_names``) and be overwritten at the
    START of every ``run_turn``. So while session ``a``'s turn was mid-flight
    (its tool-gate not yet fired), a concurrent session ``b`` turn would
    overwrite that shared set with ``b``'s names — and ``a``'s hook would then
    consult ``b``'s set, skipping or gating the wrong tools.

    Deterministic interleave: ``a`` bridges ``shared_tool`` and blocks after one
    delta, BEFORE its gate fires. While ``a`` is parked, ``b`` (which bridges a
    DISJOINT set NOT containing ``shared_tool``) runs to completion. Then ``a``
    is released and gates ``shared_tool``. Correct (closure-bound) behaviour:
    ``a``'s hook still has ``{shared_tool}`` and SKIPS it, so it runs despite the
    DENY verdict. The old shared-state code would have ``a``'s hook read ``b``'s
    set, gate ``shared_tool``, and wrongly DENY it.
    """
    # DENY everything that reaches the evaluator — so "skipped" (ran) vs "gated"
    # (denied) is an unambiguous signal of which set the hook used.
    deny = _FakePolicyVerdict("POLICY_ACTION_DENY", reason="should not be consulted for a")
    evaluator = _FakePolicyEvaluator({"PHASE_TOOL_CALL": deny})

    gate = asyncio.Event()
    a_call = _FakeSDKToolCall("shared_tool", {"x": 1}, call_id="a1")
    captured: dict[str, Any] = {}

    class _FakeHooks:
        PostToolCallHook = _FakePostToolCallHook
        PreToolCallDecideHook = _FakePreToolCallDecideHook

    class _FakeTypes:
        AntigravityCancelledError = _AntigravityCancelledError
        HookResult = _FakeHookResult
        ToolCall = _FakeSDKToolCall
        BuiltinTools = _BuiltinTools

    # Session ``b`` uses the ordinary scripted conversation; session ``a`` uses
    # the blocking, gate-driven one. They are told apart by the bridged tool
    # name baked into the agent config the executor builds.
    b_queue: collections.deque[list[_TurnAction]] = collections.deque(
        [[_exec_tool("beta", {"y": 2}, call_id="b1")]]
    )

    class _SwitchingModule:
        LocalAgentConfig = _FakeLocalAgentConfig
        hooks = _FakeHooks
        types = _FakeTypes

        @staticmethod
        def Agent(config: Any) -> Any:
            tool_names = {
                getattr(t, "__name__", "") for t in (getattr(config, "tools", None) or [])
            }
            if "shared_tool" in tool_names:
                conv = _GatingBlockingConversation(
                    list(getattr(config, "hooks", []) or []), gate, a_call
                )
                captured["conv_a"] = conv
            else:
                conv = _FakeConversation(list(getattr(config, "hooks", []) or []), b_queue)
                captured["conv_b"] = conv

            class _Agent:
                def __init__(self) -> None:
                    self._conversation = conv

                @property
                def conversation(self) -> Any:
                    return self._conversation

                async def __aenter__(self) -> Any:
                    return self

                async def __aexit__(self, *_a: object) -> None:
                    return None

            return _Agent()

    monkeypatch.setattr(ag, "_ensure_antigravity_sdk", lambda: _SwitchingModule())

    executor = AntigravityExecutor()
    executor._policy_evaluator = evaluator
    executor._tool_executor = lambda name, args: _async_ok()

    tools_a = [{"name": "shared_tool", "description": "bridged for a", "parameters": {}}]
    tools_b = [{"name": "beta", "description": "bridged for b", "parameters": {}}]

    # Start ``a``; it streams one delta then parks on the gate (mid-flight).
    a_collected: list[Any] = []
    a_started = asyncio.Event()

    async def _drive_a() -> None:
        async for event in executor.run_turn(
            [{"role": "user", "content": "a", "session_id": "a"}],
            tools=tools_a,
            system_prompt="sys",
        ):
            a_collected.append(event)
            if isinstance(event, TextChunk):
                a_started.set()

    a_task = asyncio.create_task(_drive_a())
    await asyncio.wait_for(a_started.wait(), timeout=5)

    # With ``a`` provably parked before its gate, run ``b`` to completion. Under
    # the old code this is the write that would clobber the shared skip set.
    await _drain(executor, [{"role": "user", "content": "b", "session_id": "b"}], tools_b)

    # Release ``a``; its tool-gate now fires and must use ``a``'s own set.
    gate.set()
    await asyncio.wait_for(a_task, timeout=5)

    conv_a = captured["conv_a"]
    conv_b = captured["conv_b"]
    # ``a`` bridged ``shared_tool`` → its hook skipped it → it ran, despite the
    # DENY. A regression to shared state would deny it (a's hook seeing b's set).
    assert conv_a.executed_tools == ["shared_tool"]
    assert conv_a.denied_tools == []
    # ``b`` bridged ``beta`` → skipped → ran. Sanity that the interleaved turn
    # behaved normally.
    assert conv_b.executed_tools == ["beta"]
    assert conv_b.denied_tools == []
    # The evaluator was never consulted for ``shared_tool`` or ``beta`` — both
    # were correctly skipped as bridged for their own session.
    gated_names = {d["name"] for (p, d) in evaluator.calls if p == "PHASE_TOOL_CALL"}
    assert gated_names == set()
