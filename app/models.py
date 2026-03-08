"""Domain models for the Colab proof-of-concept app."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeSummary:
    """Structured runtime metadata used for table and JSON output."""

    message: str
    timestamp_utc: str
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the runtime summary to a JSON-friendly dictionary."""
        return asdict(self)
