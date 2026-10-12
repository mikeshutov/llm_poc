from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCachePolicy:
    cache_ttl_seconds: int
    is_user_specific: bool = False

    def __post_init__(self) -> None:
        if self.cache_ttl_seconds <= 0:
            raise ValueError("Tool cache policy cache_ttl_seconds must be positive.")

    def is_cache_allowed_for_user(self, user_id: str | None) -> bool:
        """Return whether this policy may cache for the current identity."""
        return not self.is_user_specific or bool(user_id and user_id.strip())
