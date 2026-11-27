from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest


from agent_runtime.context import bind_agent_context
from agent_runtime.models.agent_profile import AgentKind
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from personalization.profile.models import UserProfile
from capabilities.registry import global_registry
from common.http import HttpClientError
from requests.exceptions import Timeout
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY, ToolRegistry, ToolRegistryError, UnknownToolError, register_default_tools
from common.cache.policy import ToolCachePolicy
from capabilities.registry.models import RateLimitPolicy, Tool
from capabilities.retry.policy import RetryPolicy
from retrieval.evidence import EvidenceView, ToolResult
from capabilities.adapters.profile import USER_MANAGEMENT_TOOL


def test_runtime_structured_tool_without_cache_policy_is_executable() -> None:
    registry = ToolRegistry(enable_cache=True)
    registry.register(USER_MANAGEMENT_TOOL)

    result = registry.call_tool(USER_MANAGEMENT_TOOL.name, {"task": "Remember that I like tea."})

    assert result.result["error"] == "User management is unavailable outside an active request."


def test_registry_rejects_overwriting_an_existing_tool(tool_factory) -> None:
    registry = ToolRegistry()
    original = tool_factory("shared", lambda args: "original")
    registry.register(original)
    registry.register(original)

    with pytest.raises(ToolRegistryError, match="already registered"):
        registry.register(tool_factory("shared", lambda args: "replacement"))

    assert registry.call_tool("shared") == "original"


def test_same_named_user_tools_are_isolated_and_cannot_replace_builtins(
    tool_factory,
    agent_state_factory,
) -> None:
    registry = ToolRegistry()
    registry.register(tool_factory("assistant", lambda args: "Alice"), user_id="alice")
    registry.register(tool_factory("assistant", lambda args: "Bob"), user_id="bob")
    registry.register(tool_factory("builtin", lambda args: "shared"))

    for owner, expected in [("alice", "Alice"), ("bob", "Bob")]:
        with bind_agent_context(agent_state_factory(
            execution_context=AgentExecutionContext.new(user_profile=UserProfile(user_id=owner)),
        )):
            assert registry.call_tool("assistant") == expected
            assert registry.call_tool("builtin") == "shared"

    with pytest.raises(ToolRegistryError, match="reserved"):
        registry.register(tool_factory("builtin", lambda args: "replacement"), user_id="alice")


def test_agent_registration_is_idempotent_and_preserves_other_users(
    agent_profile_factory,
    registered_tool_registry,
) -> None:
    registry = registered_tool_registry
    original = registry.get("coding")
    global_registry.register_default_tools()
    assert registry.get("coding") is original

    alice = agent_profile_factory(
        name="assistant",
        scope="main_agent",
        kind=AgentKind.USER_AGENT,
        description="Alice's agent",
    )
    bob = agent_profile_factory(
        name="assistant",
        scope="main_agent",
        kind=AgentKind.USER_AGENT,
        description="Bob's agent",
    )
    global_registry.register_agent_tools([alice], user_id="alice")
    alice_tool = registry.get("assistant", user_id="alice")
    global_registry.register_agent_tools([alice], user_id="alice")
    global_registry.register_agent_tools([bob], user_id="bob")

    assert registry.get("assistant", user_id="alice") is alice_tool
    assert registry.get("assistant", user_id="bob").description == "Bob's agent"
    with pytest.raises(ToolRegistryError, match="reserved"):
        global_registry.register_agent_tools([
            agent_profile_factory(name="coding", scope="main_agent", kind=AgentKind.USER_AGENT),
        ], user_id="alice")


@pytest.mark.parametrize("iteration", range(2))
def test_registered_tools_do_not_leak_between_tests(tool_factory, registered_tool_registry, iteration):
    with pytest.raises(UnknownToolError):
        registered_tool_registry.get("test_only_tool")

    registered_tool_registry.register(tool_factory("test_only_tool", lambda args: iteration))
    assert registered_tool_registry.call_tool("test_only_tool") == iteration


def test_tool_registry_serializes_calls_for_shared_rate_limit_key(tool_factory) -> None:
    registry = ToolRegistry()
    call_times: list[tuple[str, float]] = []
    lock = threading.Lock()

    def make_callback(name: str):
        def _callback(_tool_input):
            with lock:
                call_times.append((name, time.monotonic()))
            return {"name": name}

        return _callback

    policy = RateLimitPolicy(max_requests=1, window_seconds=0.05)
    registry.register(Tool(tool_factory("news_search", make_callback("news_search")), rate_limit_key="brave", rate_limit_policy=policy))
    registry.register(Tool(tool_factory("generic_web_search", make_callback("generic_web_search")), rate_limit_key="brave", rate_limit_policy=policy))

    thread_one = threading.Thread(target=registry.call_tool, args=("news_search", {}))
    thread_two = threading.Thread(target=registry.call_tool, args=("generic_web_search", {}))
    thread_one.start()
    thread_two.start()
    thread_one.join()
    thread_two.join()

    assert len(call_times) == 2
    assert abs(call_times[1][1] - call_times[0][1]) >= 0.045


def test_tool_registry_retries_timeout_when_policy_allows(tool_factory) -> None:
    registry = ToolRegistry()
    attempts = 0

    def callback(_tool_input):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise Timeout("timed out")
        return {"ok": True}

    registry.register(
        Tool(
            tool_factory("retry_tool", callback),
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
        )
    )

    result = registry.call_tool("retry_tool", {})

    assert result == {"ok": True}
    assert attempts == 2


def test_tool_registry_retries_http_429_when_policy_allows(tool_factory) -> None:
    registry = ToolRegistry()
    attempts = 0

    def callback(_tool_input):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HttpClientError("HTTP 429 on https://example.com: rate limited")
        return {"ok": True}

    registry.register(
        Tool(
            tool_factory("retry_tool", callback),
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
        )
    )

    result = registry.call_tool("retry_tool", {})

    assert result == {"ok": True}
    assert attempts == 2


def test_tool_registry_retries_wrapped_http_429_when_policy_allows(tool_factory) -> None:
    registry = ToolRegistry()
    attempts = 0

    def callback(_tool_input):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("HTTP 429 on https://api.search.brave.com/res/v1/web/search: rate limited")
        return {"ok": True}

    registry.register(
        Tool(
            tool_factory("retry_tool", callback),
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
        )
    )

    result = registry.call_tool("retry_tool", {})

    assert result == {"ok": True}
    assert attempts == 2


def test_tool_registry_cache_miss_invokes_tool_and_stores_result(
    cache_repository_factory,
    tool_factory,
) -> None:
    cache = cache_repository_factory()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(tool_factory("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2))

    result = registry.call_tool("cache_tool", {"query": "chicken"})

    assert result == {"value": 1}
    assert calls == 1
    assert cache.get_calls == 1
    assert cache.put_calls == 1


def test_tool_registry_cache_hit_skips_tool_invocation(cache_repository_factory, tool_factory) -> None:
    cache = cache_repository_factory()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(tool_factory("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2))

    first = registry.call_tool("cache_tool", {"query": "chicken"})
    second = registry.call_tool("cache_tool", {"query": "chicken"})

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert calls == 1
    assert cache.get_calls == 2
    assert cache.put_calls == 1


def test_search_meals_cache_hit_reconstructs_tool_result_with_evidence(
    cache_repository_factory,
    tool_factory,
) -> None:
    cache = cache_repository_factory()
    calls = 0
    evidence = EvidenceView(
        summary="A chicken soup recipe.",
        source="search_meals",
        type="meal_results",
        data={
            "item_id": "meal-1",
            "title": "Chicken soup",
        })

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return ToolResult(result={"name": "Chicken soup"}, evidence=[evidence])

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(tool_factory("search_meals", callback), cache_policy=ToolCachePolicy(60), version=2))

    first = registry.call_tool("search_meals", {"query": "chicken"})
    second = registry.call_tool("search_meals", {"query": "chicken"})

    assert isinstance(first, ToolResult)
    assert isinstance(second, ToolResult)
    assert second.evidence[0].data["title"] == "Chicken soup"
    assert calls == 1


def test_tool_registry_legacy_cache_payload_is_treated_as_miss(
    cache_repository_factory,
    tool_factory,
) -> None:
    cache = cache_repository_factory()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return ToolResult(result={"value": calls}, evidence=[])

    tool = Tool(tool_factory("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2)
    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(tool)
    cache_key = cache.build_cache_key(
        tool_name=tool.name,
        tool_version=tool.version,
        tool_input={"query": "chicken"},
    )
    cache.entries[cache_key] = {"result": {"value": 0}, "evidence": []}

    result = registry.call_tool(tool.name, {"query": "chicken"})

    assert isinstance(result, ToolResult)
    assert result.result == {"value": 1}
    assert calls == 1


def test_tool_registry_malformed_cache_envelope_is_treated_as_miss(
    cache_repository_factory,
    tool_factory,
) -> None:
    cache = cache_repository_factory()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    tool = Tool(tool_factory("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2)
    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(tool)
    cache_key = cache.build_cache_key(
        tool_name=tool.name,
        tool_version=tool.version,
        tool_input={"query": "chicken"},
    )
    cache.entries[cache_key] = {
        "cache_payload_version": 1,
        "output_kind": "tool_result",
        "output": {"result": {"value": 0}, "evidence": ["invalid"]},
    }

    assert registry.call_tool(tool.name, {"query": "chicken"}) == {"value": 1}
    assert calls == 1


def test_tool_registry_cache_lookup_error_falls_back_to_tool(cache_repository_factory, tool_factory) -> None:
    cache = cache_repository_factory(fail_get=True)
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(tool_factory("cache_tool", callback), cache_policy=ToolCachePolicy(60)))

    assert registry.call_tool("cache_tool", {"query": "chicken"}) == {"value": 1}
    assert calls == 1


def test_tool_registry_cache_store_error_does_not_fail_tool_request(
    cache_repository_factory,
    tool_factory,
) -> None:
    cache = cache_repository_factory(fail_put=True)
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(tool_factory("cache_tool", callback), cache_policy=ToolCachePolicy(60)))

    assert registry.call_tool("cache_tool", {"query": "chicken"}) == {"value": 1}
    assert calls == 1


def test_user_specific_cache_isolated_by_user(cache_repository_factory, tool_factory) -> None:
    cache = cache_repository_factory()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(
        Tool(tool_factory("private_search", callback), cache_policy=ToolCachePolicy(60, is_user_specific=True))
    )

    with patch("capabilities.registry.global_registry.get_current_user_id", return_value="user-a"):
        first = registry.call_tool("private_search", {"query": "chicken"})
        second = registry.call_tool("private_search", {"query": "chicken"})
    with patch("capabilities.registry.global_registry.get_current_user_id", return_value="user-b"):
        third = registry.call_tool("private_search", {"query": "chicken"})

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert third == {"value": 2}
    assert calls == 2
    assert cache.put_calls == 2


def test_user_specific_cache_bypasses_anonymous_requests(cache_repository_factory, tool_factory) -> None:
    cache = cache_repository_factory()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(
        Tool(tool_factory("private_search", callback), cache_policy=ToolCachePolicy(60, is_user_specific=True))
    )

    with patch("capabilities.registry.global_registry.get_current_user_id", return_value=None):
        first = registry.call_tool("private_search", {"query": "chicken"})
        second = registry.call_tool("private_search", {"query": "chicken"})

    assert first == {"value": 1}
    assert second == {"value": 2}
    assert calls == 2
    assert cache.get_calls == 0
    assert cache.put_calls == 0


def test_cache_policy_controls_cacheability() -> None:
    assert ToolCachePolicy(60).is_cache_allowed_for_user(None) is True
    assert ToolCachePolicy(60, is_user_specific=True).is_cache_allowed_for_user("user-1") is True
    assert ToolCachePolicy(60, is_user_specific=True).is_cache_allowed_for_user(None) is False
    with pytest.raises(ValueError, match="cache_ttl_seconds must be positive"):
        ToolCachePolicy(0)

