"""E2E test: web research workflow user journey.

Exercises the realistic research flow where an agent searches the web,
receives results, and retains context across a follow-up turn:

1. Start a local HTTP stub server returning canned Perplexity results
   with a distinctive sentinel string.
2. Set ``OMNIGENT_PERPLEXITY_BASE_URL`` to point at the stub.
3. Upload an agent with ``web_search`` (Perplexity provider).
4. **Turn 1**: Ask the agent to search for information — verify the
   ``web_search`` function_call appears in output and the stub's
   sentinel text surfaces in the response.
5. **Turn 2**: Ask a follow-up referencing the first answer — verify
   the agent retains context from turn 1.

The stub avoids any dependency on a real Perplexity API key. The pattern
mirrors ``test_web_search_async_dispatch_e2e.py``.

Usage::

    pytest tests/e2e/test_journey_web_research.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import (
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
    upload_agent,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_SEARCH_TEST_DIR = _REPO_ROOT / "tests" / "resources" / "agents" / "web-search-test"

_STUB_SENTINEL = "The omnigent framework was created in 2024 by a distributed team."
_FAKE_BEARER = "test-bearer-journey-not-validated"


# ---------------------------------------------------------------------------
# Stub Perplexity server
# ---------------------------------------------------------------------------


class _FakePerplexityHandler(BaseHTTPRequestHandler):
    """Return a canned chat-completion with the sentinel."""

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [{"message": {"content": _STUB_SENTINEL}}],
                "citations": ["https://example.invalid/omnigent-creation"],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# Module-level: the env var must be set before the session-scoped
# live_server fixture reads os.environ.
_FAKE_SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _FakePerplexityHandler)
threading.Thread(target=_FAKE_SERVER.serve_forever, daemon=True).start()
os.environ["OMNIGENT_PERPLEXITY_BASE_URL"] = (
    f"http://127.0.0.1:{_FAKE_SERVER.server_address[1]}/chat/completions"
)


# ---------------------------------------------------------------------------
# Agent fixture
# ---------------------------------------------------------------------------


def _materialize_with_resolved_env(src: Path, dst: Path) -> Path:
    """Copy the fixture agent and resolve ``${VAR}`` references client-side.

    The server does not expand env vars in uploaded bundles. Real clients
    resolve them before upload, so this fixture mirrors that.
    """
    shutil.copytree(src, dst)
    cfg = dst / "config.yaml"
    resolved = (
        cfg.read_text()
        .replace("${PERPLEXITY_API_KEY}", _FAKE_BEARER)
        .replace("${OPENAI_API_KEY}", os.environ.get("OPENAI_API_KEY", _FAKE_BEARER))
    )
    cfg.write_text(resolved)
    return dst


@pytest.fixture(scope="session")
def web_research_agent(
    http_client: httpx.Client,
    databricks_workspace_host: str | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Upload a web-search agent backed by the stub Perplexity server."""
    staging = tmp_path_factory.mktemp("web-research-journey-bundle")
    prepared = _materialize_with_resolved_env(_WEB_SEARCH_TEST_DIR, staging / "agent")
    return upload_agent(
        http_client,
        prepared,
        rewrite_model_for_databricks=databricks_workspace_host is not None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _function_calls(body: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Return all ``function_call`` output items matching *name*."""
    return [
        item
        for item in body.get("output", [])
        if item.get("type") == "function_call" and item.get("name") == name
    ]


def _all_text(body: dict[str, Any]) -> str:
    """Concatenate every text block from message items in the output."""
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            text = block.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.llm_flaky(reruns=2)
def test_web_research_workflow(
    http_client: httpx.Client,
    web_research_agent: str,
    live_runner_id: str,
) -> None:
    """Agent searches → receives stub result → retains context on follow-up.

    Turn 1: ask the agent to search for information about the omnigent
    framework creation date. Verify:
    - A ``web_search`` function_call item appears in the output.
    - The stub's sentinel text surfaces in the response text.

    Turn 2: ask a follow-up question that requires context from turn 1.
    Verify:
    - The agent references the year "2024" from the stub's answer,
      demonstrating context retention across turns.
    """
    session_id = create_runner_bound_session(
        http_client, agent_name=web_research_agent, runner_id=live_runner_id
    )

    # ── Turn 1: search ──────────────────────────────────────
    response_id_1 = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Search the web for information about the omnigent framework "
            "creation date. Quote the search result verbatim."
        ),
    )
    body_1 = poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id_1,
        timeout=300,
    )

    assert body_1["status"] == "completed", (
        f"Turn 1 status={body_1['status']!r}, output={body_1.get('output', [])}"
    )

    # The async-dispatch path emits a function_call item for web_search.
    assert _function_calls(body_1, "web_search"), (
        "No web_search function_call in turn 1 output. "
        f"Output types: {[i.get('type') for i in body_1.get('output', [])]}"
    )

    # The stub's sentinel should surface via the async_work_complete
    # drain (as user input_text) or in the assistant's final text.
    turn_1_text = _all_text(body_1)
    assert _STUB_SENTINEL in turn_1_text or "2024" in turn_1_text, (
        f"Stub sentinel or year '2024' missing from turn 1 text. "
        f"Text (first 500 chars): {turn_1_text[:500]!r}"
    )

    # ── Turn 2: follow-up requiring context retention ───────
    response_id_2 = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content="Based on what you found, when was the omnigent framework created?",
    )
    body_2 = poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id_2,
        timeout=300,
    )

    assert body_2["status"] == "completed", (
        f"Turn 2 status={body_2['status']!r}, output={body_2.get('output', [])}"
    )

    turn_2_text = _all_text(body_2)
    assert "2024" in turn_2_text, (
        "Turn 2 did not reference '2024' from turn 1's search result — "
        f"context retention failed. Text: {turn_2_text[:500]!r}"
    )
