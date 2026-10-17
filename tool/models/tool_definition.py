from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from tool.cache.policy import ToolCachePolicy
from tool.retry.policy import RetryPolicy

DEFAULT_RESULT_FIELDS = (
    "summary",
    "url",
    "image_url",
    "published_at",
    "source",
    "entity_type",
    "location_name",
    "metadata",
)


@dataclass(frozen=True)
class RateLimitPolicy:
    max_requests: int
    window_seconds: float


@dataclass(frozen=True)
class Tool:
    fn: Any
    result_type: str = "generic"
    excluded_result_fields: frozenset[str] = frozenset()
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
        description = getattr(self.fn, "description", "")
        fields = self.available_result_fields
        if not fields:
            return description
        return f"{description}\nReturns entities with fields: {', '.join(fields)}".strip()

    @property
    def available_result_fields(self) -> tuple[str, ...]:
        from tool.result_schemas import RESULT_SCHEMAS

        result_schema = RESULT_SCHEMAS.get(self.result_type)
        fields = (
            tuple(result_schema.model_fields)
            if result_schema is not None
            else DEFAULT_RESULT_FIELDS
        )
        return tuple(
            name
            for name in fields
            if name not in self.excluded_result_fields
        )

    @property
    def args_schema(self) -> Any:
        return getattr(self.fn, "args_schema", None)

    def invoke(self, tool_input: Any = None) -> Any:
        return self.fn.invoke(tool_input or {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fn, name)
