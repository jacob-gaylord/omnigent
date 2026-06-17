"""Tests for Codex-native elicitation correlation helpers."""

from __future__ import annotations

import re

import pytest

from omnigent.codex_native_elicitation import (
    codex_elicitation_id,
    is_codex_request_id,
)


@pytest.mark.parametrize("value", [0, 1, -1, "", "req_abc"])
def test_is_codex_request_id_accepts_ints_and_strings(value: int | str) -> None:
    """
    Codex JSON-RPC ids may be integers or strings.

    :param value: Candidate request id.
    :returns: None.
    """
    assert is_codex_request_id(value) is True


@pytest.mark.parametrize(
    "value",
    [True, False, None, 1.5, [], {}],
)
def test_is_codex_request_id_rejects_unsupported_values(value: object) -> None:
    """
    Booleans are rejected even though ``bool`` subclasses ``int``.

    :param value: Candidate request id.
    :returns: None.
    """
    assert is_codex_request_id(value) is False


def test_codex_elicitation_id_is_deterministic() -> None:
    """
    Repeating the same app-server request produces the same card id.

    :returns: None.
    """
    first_id = codex_elicitation_id(
        "conv_abc123",
        "item/tool/requestUserInput",
        12,
    )
    second_id = codex_elicitation_id(
        "conv_abc123",
        "item/tool/requestUserInput",
        12,
    )

    assert second_id == first_id


@pytest.mark.parametrize(
    ("session_id", "method", "request_id"),
    [
        ("conv_other", "item/tool/requestUserInput", 12),
        ("conv_abc123", "item/tool/other", 12),
        ("conv_abc123", "item/tool/requestUserInput", 13),
    ],
)
def test_codex_elicitation_id_changes_with_correlation_inputs(
    session_id: str,
    method: str,
    request_id: int | str,
) -> None:
    """
    Each correlation field contributes to the stable id digest.

    :param session_id: Omnigent session id.
    :param method: Codex app-server method.
    :param request_id: Codex JSON-RPC request id.
    :returns: None.
    """
    baseline_id = codex_elicitation_id(
        "conv_abc123",
        "item/tool/requestUserInput",
        12,
    )

    assert codex_elicitation_id(session_id, method, request_id) != baseline_id


def test_codex_elicitation_id_uses_stable_prefix_and_digest_format() -> None:
    """
    Elicitation ids include the Codex prefix and a 32-character hex digest.

    :returns: None.
    """
    elicitation_id = codex_elicitation_id(
        "conv_abc123",
        "item/tool/requestUserInput",
        "req_abc",
    )

    assert elicitation_id.startswith("elicit_codex_")
    assert re.fullmatch(r"elicit_codex_[0-9a-f]{32}", elicitation_id) is not None


def test_codex_elicitation_id_distinguishes_int_and_string_request_ids() -> None:
    """
    Numeric and string JSON values remain distinct in the hashed payload.

    :returns: None.
    """
    int_id = codex_elicitation_id(
        "conv_abc123",
        "item/tool/requestUserInput",
        12,
    )
    string_id = codex_elicitation_id(
        "conv_abc123",
        "item/tool/requestUserInput",
        "12",
    )

    assert string_id != int_id
