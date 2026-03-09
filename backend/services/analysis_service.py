"""Analysis orchestration service.

Most analysis logic is implemented directly in the analysis router
(backend/api/routers/analysis.py) to keep things simple. This module
provides shared utilities that the router uses.
"""
from __future__ import annotations

from datetime import datetime, timezone


def timestamp_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
