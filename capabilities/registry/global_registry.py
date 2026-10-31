from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic, sleep
from typing import Any

from common.http import HttpClientError
from common.logging import record_node_log
from agent_runtime.constants import NodeName
from capabilities.registry.tools import tools
from capabilities.discovery.discover_capabilities_tool import CAPABILITY_DISCOVERY_TOOL
from capabilities.adapters.profile import USER_MANAGEMENT_TOOL
from common.cache.repository import InvalidCachePayloadError, ToolCacheRepository
from agent_runtime.context import get_current_user_id
from requests.exceptions import Timeout


class ToolRegistryError(Exception):
    pass


class UnknownToolError(ToolRegistryError):
    pass


class DisallowedToolError(ToolRegistryError):
    pass


def _record_tool_registry_log(message: str, payload: dict[str, Any], *, level: str = "info", stage: str = "") -> None:
    record_node_log(
        node_name=NodeName.EXECUTOR,
        stage=stage or "registry",
        event_type="tool_registry",
        level=level,
        message=message,
        payload=payload,
    )


@dataclass
class _RateLimitState:
    lock: Lock = field(default_factory=Lock)
    request_timestamps: deque[float] = field(default_factory=deque)


class ToolRegistry:
    def __init__(self, *, cache_repository=None, enable_cache: bool = False) -> None:
        self._tools: dict[str, Any] = {}
        self._rate_limit_states: dict[str, _RateLimitState] = defaultdict(_RateLimitState)
        self._cache_repository = cache_repository
        self._enable_cache = enable_cache

    def register(self, tool: Any) -> None:
        name = getattr(tool, "name", None)
        invoke = getattr(tool, "invoke", None)
        if not isinstance(name, str) or not name:
            raise ToolRegistryError("Tool must define a non-empty string name.")
        if not callable(invoke):
            raise ToolRegistryError(f"Tool '{name}' must expose an invoke method.")
        self._tools[name] = tool

    def get(self, name: str) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool '{name}'.")
        return tool

    def _acquire_rate_limit_slot(self, tool: Any) -> None:
        rate_limit_key = getattr(tool, "rate_limit_key", None)
        rate_limit_policy = getattr(tool, "rate_limit_policy", None)
        if not rate_limit_key or rate_limit_policy is None:
            return

        state = self._rate_limit_states[rate_limit_key]
        while True:
            with state.lock:
                now = monotonic()
                cutoff = now - rate_limit_policy.window_seconds
                while state.request_timestamps and state.request_timestamps[0] <= cutoff:
                    state.request_timestamps.popleft()

                if len(state.request_timestamps) < rate_limit_policy.max_requests:
                    state.request_timestamps.append(now)
                    return

                wait_seconds = max(0.0, state.request_timestamps[0] + rate_limit_policy.window_seconds - now)

            if wait_seconds > 0:
                sleep(wait_seconds)

    @staticmethod
    def _should_retry(exc: Exception, tool: Any) -> bool:
        retry_policy = getattr(tool, "retry_policy", None)
        if retry_policy is None:
            return False
        if retry_policy.retry_on_timeout and isinstance(exc, Timeout):
            return True
        message = str(exc)
        if isinstance(exc, HttpClientError) or "HTTP " in message:
            if retry_policy.retry_on_429 and "HTTP 429" in message:
                return True
            if retry_policy.retry_on_5xx and any(f"HTTP {status}" in message for status in range(500, 600)):
                return True
        return False

    def call_tool(self, name: str, tool_input: Any = None, *, allowed_tool_names: set[str] | None = None) -> Any:
        if allowed_tool_names is not None and name not in allowed_tool_names:
            raise DisallowedToolError(f"Tool '{name}' is not allowed for this agent.")
        tool = self.get(name)
        # Runtime-only LangChain tools do not carry the application Tool
        # wrapper's cache_policy attribute. They remain executable, simply
        # without registry caching.
        cache_policy = getattr(tool, "cache_policy", None) if self._enable_cache else None
        cache_ttl_seconds = cache_policy.cache_ttl_seconds if cache_policy is not None else 0
        cache_repository = None
        cache_user_id = get_current_user_id() if cache_policy and cache_policy.is_user_specific else None
        if cache_policy is not None and cache_policy.is_cache_allowed_for_user(cache_user_id):
            try:
                cache_repository = self._cache_repository or ToolCacheRepository()
                cached_output = cache_repository.get(
                    tool=tool,
                    tool_input=tool_input,
                    user_id=cache_user_id,
                )
                if cached_output is not None:
                    _record_tool_registry_log("Tool cache hit", {"tool": name}, stage="cache")
                    return cached_output
            except InvalidCachePayloadError as exc:
                _record_tool_registry_log(
                    "Tool cache payload invalid",
                    {
                        "tool": name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    level="warning",
                    stage="cache",
                )
            except Exception as exc:
                _record_tool_registry_log(
                    "Tool cache lookup failed",
                    {
                        "tool": name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    level="warning",
                    stage="cache",
                )
                cache_repository = None
        retry_policy = getattr(tool, "retry_policy", None)
        max_attempts = max(1, getattr(retry_policy, "max_attempts", 1))
        try:
            for attempt in range(1, max_attempts + 1):
                self._acquire_rate_limit_slot(tool)
                try:
                    output = tool.invoke(tool_input or {})
                    if cache_repository is not None:
                        try:
                            cache_repository.put(
                                tool=tool,
                                tool_input=tool_input,
                                output_payload=output,
                                cache_ttl_seconds=cache_ttl_seconds,
                                user_id=cache_user_id,
                            )
                        except Exception as exc:
                            _record_tool_registry_log(
                                "Tool cache store failed",
                                {
                                    "tool": name,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                                level="warning",
                                stage="cache",
                            )
                    return output
                except Exception as exc:
                    should_retry = attempt < max_attempts and self._should_retry(exc, tool)
                    if not should_retry:
                        raise
                    backoff_seconds = max(0.0, getattr(retry_policy, "backoff_seconds", 0.0))
                    if backoff_seconds > 0:
                        sleep(backoff_seconds)
        except Exception as exc:
            _record_tool_registry_log(
                "Tool invocation failed",
                {
                    "tool": name,
                    "tool_input": tool_input,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                level="error",
                stage="invoke",
            )
            raise


GLOBAL_TOOL_REGISTRY = ToolRegistry(enable_cache=True)
_REGISTERED_DEFAULTS = False


def register_default_tools() -> None:
    global _REGISTERED_DEFAULTS
    if _REGISTERED_DEFAULTS:
        return
    for tool in tools:
        GLOBAL_TOOL_REGISTRY.register(tool)
    # Runtime-only orchestration capability. It is intentionally not part of
    # ``capabilities.registry.tools`` and therefore is never persisted by the tool seeder.
    GLOBAL_TOOL_REGISTRY.register(CAPABILITY_DISCOVERY_TOOL)
    GLOBAL_TOOL_REGISTRY.register(USER_MANAGEMENT_TOOL)
    _REGISTERED_DEFAULTS = True


def call_tool(name: str, tool_input: Any = None, allowed_tool_names: set[str] | None = None) -> Any:
    register_default_tools()
    return GLOBAL_TOOL_REGISTRY.call_tool(name=name, tool_input=tool_input, allowed_tool_names=allowed_tool_names)


__all__ = [
    "DisallowedToolError",
    "GLOBAL_TOOL_REGISTRY",
    "ToolRegistryError",
    "UnknownToolError",
    "register_default_tools",
    "call_tool",
]
