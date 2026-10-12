from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tool.cache.policy import ToolCachePolicy
from tool.retry.policy import RetryPolicy


@dataclass(frozen=True)
class RateLimitPolicy:
    max_requests: int
    window_seconds: float


@dataclass(frozen=True)
class Tool:
    fn: Any
    result_type: str = "generic"
    rate_limit_key: str | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limit_policy: RateLimitPolicy | None = None
    version: int = 1
    cache_policy: ToolCachePolicy | None = None

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
