from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCachePolicy:
    ttl_seconds: int
    user_specific: bool = False

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("Tool cache policy ttl_seconds must be positive.")

    def allows(self, user_id: str | None) -> bool:
        """Return whether this policy may cache for the current identity."""
        return not self.user_specific or bool(user_id and user_id.strip())
