"""Service layer for collecting runtime information."""

from __future__ import annotations

import platform
from datetime import datetime, timezone

from app.models import RuntimeSummary


def build_runtime_summary(user_name: str) -> RuntimeSummary:
    """Build a structured runtime summary for display and export."""
    return RuntimeSummary(
        message=f"Hello {user_name}!",
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
