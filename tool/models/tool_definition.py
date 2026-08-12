from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retry_on_timeout: bool = True
    retry_on_429: bool = True
    retry_on_5xx: bool = True
    backoff_seconds: float = 1.0


@dataclass(frozen=True)
class RateLimitPolicy:
    max_requests: int
    window_seconds: float


@dataclass(frozen=True)
class Tool:
    fn: Any
    rate_limit_key: str | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limit_policy: RateLimitPolicy | None = None

    @property
    def name(self) -> str:
        return getattr(self.fn, "name")

    @property
    def description(self) -> str:
        return getattr(self.fn, "description", "")

    @property
    def args_schema(self) -> Any:
        return getattr(self.fn, "args_schema", None)

    def invoke(self, tool_input: Any = None) -> Any:
        return self.fn.invoke(tool_input or {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fn, name)
