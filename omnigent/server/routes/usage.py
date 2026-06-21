"""Read-only route for live subscription usage.

Exposes ``GET /v1/usage`` so the web UI can show current Claude + Codex
subscription limit usage (5-hour + weekly windows) for the logged-in user's
subscriptions.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request

from omnigent.server.auth import AuthProvider
from omnigent.server.routes._auth_helpers import require_user
from omnigent.usage import get_usage


def create_usage_router(auth_provider: AuthProvider | None = None) -> APIRouter:
    """Build the usage router."""
    router = APIRouter()

    @router.get("/usage")
    async def usage(request: Request) -> dict[str, Any]:
        """Return current Claude + Codex subscription usage (cached ~60s)."""
        require_user(request, auth_provider)
        return await asyncio.to_thread(get_usage)

    return router
