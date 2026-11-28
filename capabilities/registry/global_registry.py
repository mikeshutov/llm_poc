from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock, RLock
from time import monotonic, sleep
from typing import Any

from langchain_core.tools import StructuredTool

from common.http import HttpClientError
from common.logging import record_node_log
from agent_runtime.constants import NodeName
from capabilities.registry.tools import tools, WORKSPACE_MUTATION_TOOLS, WORKING_MEMORY_TOOLS
from capabilities.discovery.discover_capabilities_tool import CAPABILITY_DISCOVERY_TOOL
from capabilities.adapters.profile import (
    USER_MANAGEMENT_TOOL,
    update_user_names,
    update_user_tone,
)
from capabilities.adapters.user_attributes import upsert_user_attribute
from common.cache.repository import InvalidCachePayloadError, ToolCacheRepository
from agent_runtime.context import current_agent_state
from agent_runtime.models.agent_profile import AgentKind, AgentProfile
from agents.coding.profile import CODING_PROFILE
from agents.delegate.profile import DELEGATE_PROFILE
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE
from agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from agents.research.profile import RESEARCH_PROFILE
from requests.exceptions import Timeout


class ToolRegistryError(Exception):
    pass


class UnknownToolError(ToolRegistryError):
    pass


class DisallowedToolError(ToolRegistryError):
    pass


def build_agent_tool(profile):
    """Build the registry tool whose implementation runs one agent."""
    def invoke_agent(**kwargs):
        agent_state = current_agent_state.get()
        if agent_state is None:
            raise ToolRegistryError(
                f"Agent tool '{profile.name}' requires an active agent execution state."
            )
        from agent_runtime.execution.tool_executor import execute_agent_tool

        return execute_agent_tool(agent_state, profile.name, kwargs)

    return StructuredTool.from_function(
        func=invoke_agent,
        name=profile.name,
        description=profile.description,
        args_schema=profile.delegation_args_schema,
        metadata={"agent_name": profile.name, "can_run_concurrently": profile.can_run_concurrently},
    )


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
        self._tools: dict[tuple[str | None, str], Any] = {}
        self._rate_limit_states: dict[str, _RateLimitState] = defaultdict(_RateLimitState)
        self._cache_repository = cache_repository
        self._enable_cache = enable_cache

    def register(self, tool: Any, *, user_id: str | None = None, replace: bool = False) -> None:
        name = getattr(tool, "name", None)
        invoke = getattr(tool, "invoke", None)
        if not isinstance(name, str) or not name:
            raise ToolRegistryError("Tool must define a non-empty string name.")
        if not callable(invoke):
            raise ToolRegistryError(f"Tool '{name}' must expose an invoke method.")
        key = (user_id, name)
        existing = self._tools.get(key)
        if user_id is not None and (None, name) in self._tools:
            raise ToolRegistryError(f"Tool name '{name}' is reserved by a built-in capability.")
        if existing is not None and existing is not tool and not replace:
            raise ToolRegistryError(f"Tool '{name}' is already registered.")
        self._tools[key] = tool

    def get(self, name: str, *, user_id: str | None = None) -> Any:
        active_state = current_agent_state.get()
        active_user_id = active_state.execution_context.user_profile.user_id if active_state is not None else None
        owner = user_id if user_id is not None else active_user_id
        tool = self._tools.get((owner, name)) or self._tools.get((None, name))
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

    def can_run_concurrently(self, name: str) -> bool:
        """Only tools explicitly declared safe may share an execution phase."""
        try:
            tool = self.get(name)
        except UnknownToolError:
            return False
        metadata = getattr(tool, "metadata", None) or {}
        return bool(getattr(tool, "can_run_concurrently", metadata.get("can_run_concurrently", False)))

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
        active_state = current_agent_state.get()
        active_user_id = active_state.execution_context.user_profile.user_id if active_state is not None else None
        cache_user_id = active_user_id if cache_policy and cache_policy.is_user_specific else None
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
REGISTERED_AGENTS: list[AgentProfile] = [
    PROFILE_MANAGEMENT_PROFILE,
    DOCUMENT_WRITING_PROFILE,
    RESEARCH_PROFILE,
    CODING_PROFILE,
    DELEGATE_PROFILE,
]
_REGISTERED_DEFAULTS = False
_REGISTRATION_LOCK = RLock()
PROFILE_RUNTIME_TOOLS = (
    update_user_names,
    update_user_tone,
    upsert_user_attribute,
)
AGENT_ONLY_TOOLS = (*WORKSPACE_MUTATION_TOOLS, *WORKING_MEMORY_TOOLS)
AGENT_ONLY_TOOL_NAMES = {tool.name for tool in AGENT_ONLY_TOOLS}


def register_default_tools() -> None:
    global _REGISTERED_DEFAULTS
    with _REGISTRATION_LOCK:
        if _REGISTERED_DEFAULTS:
            return
        defaults = {tool.name: tool for tool in tools}
        defaults.update({
            tool.name: tool
            for tool in (CAPABILITY_DISCOVERY_TOOL, USER_MANAGEMENT_TOOL, *PROFILE_RUNTIME_TOOLS, *AGENT_ONLY_TOOLS)
        })
        for tool in defaults.values():
            GLOBAL_TOOL_REGISTRY.register(tool)
        for profile in REGISTERED_AGENTS:
            GLOBAL_TOOL_REGISTRY.register(build_agent_tool(profile))
        _REGISTERED_DEFAULTS = True


def register_agent_tools(profiles, *, user_id: str | None = None) -> None:
    with _REGISTRATION_LOCK:
        register_default_tools()
        for profile in profiles:
            owner = user_id if profile.kind == AgentKind.USER_AGENT else None
            if profile.kind == AgentKind.USER_AGENT and not owner:
                raise ToolRegistryError(f"User agent '{profile.name}' requires a user ID.")
            key = (owner, profile.name)
            existing = GLOBAL_TOOL_REGISTRY._tools.get(key)
            if existing is not None:
                metadata = getattr(existing, "metadata", None) or {}
                if metadata.get("agent_name") != profile.name:
                    raise ToolRegistryError(f"Tool '{profile.name}' is already registered as a function.")
                if owner is None or (
                    existing.description == profile.description
                    and existing.args_schema is profile.delegation_args_schema
                    and metadata.get("can_run_concurrently", False) == profile.can_run_concurrently
                ):
                    continue
            GLOBAL_TOOL_REGISTRY.register(build_agent_tool(profile), user_id=owner, replace=existing is not None)


def available_agent_tools(state) -> list:
    """Select registered agent tools available to this caller."""
    if not state.agent_profile.can_delegate_agents:
        return []
    capabilities = state.inputs.discovered_capabilities
    discovered = None if capabilities is None else {agent.agent for agent in capabilities.agents}
    defaults = state.agent_profile.default_available_agent_names
    return [
        GLOBAL_TOOL_REGISTRY.get(name, user_id=state.execution_context.user_profile.user_id)
        for name, target in (state.available_agent_states or {}).items()
        if name != state.agent_profile.name
        and target.agent_profile.delegatable
        and (discovered is None or name in discovered or name in defaults)
    ]


def call_tool(name: str, tool_input: Any = None, allowed_tool_names: set[str] | None = None) -> Any:
    register_default_tools()
    if allowed_tool_names is None and name in AGENT_ONLY_TOOL_NAMES:
        raise DisallowedToolError(f"Tool '{name}' is restricted to authorized agent profiles.")
    return GLOBAL_TOOL_REGISTRY.call_tool(name=name, tool_input=tool_input, allowed_tool_names=allowed_tool_names)


__all__ = [
    "DisallowedToolError",
    "GLOBAL_TOOL_REGISTRY",
    "REGISTERED_AGENTS",
    "ToolRegistryError",
    "UnknownToolError",
    "register_default_tools",
    "register_agent_tools",
    "build_agent_tool",
    "available_agent_tools",
    "call_tool",
]
