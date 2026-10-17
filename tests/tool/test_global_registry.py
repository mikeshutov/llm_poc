from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from common.http import HttpClientError
from requests.exceptions import Timeout
from tool.registry.global_registry import ToolRegistry
from tool.cache.repository import decode_cache_output, encode_cache_output
from tool.cache.policy import ToolCachePolicy
from tool.models import RateLimitPolicy, Tool
from tool.retry.policy import RetryPolicy
from request_orchestrator.models.evidence import EvidenceView, ToolResult


class _FakeTool:
    def __init__(self, name: str, callback) -> None:
        self.name = name
        self.description = ""
        self.args_schema = None
        self._callback = callback

    def invoke(self, tool_input=None):
        return self._callback(tool_input or {})


class _FakeCacheRepository:
    def __init__(self):
        self.entries = {}
        self.get_calls = 0
        self.put_calls = 0

    @staticmethod
    def sanitize(value):
        return value

    @staticmethod
    def build_cache_key(*, tool_name, tool_version, tool_input, user_id=None):
        return (tool_name, tool_version, user_id, repr(tool_input))

    def get(self, *, tool, tool_input, user_id=None):
        self.get_calls += 1
        cache_key = self.build_cache_key(
            tool_name=tool.name,
            tool_version=tool.version,
            tool_input=tool_input,
            user_id=user_id,
        )
        cached_output = self.entries.get(cache_key)
        return None if cached_output is None else decode_cache_output(cached_output)

    def put(self, **kwargs):
        self.put_calls += 1
        cache_key = self.build_cache_key(
            tool_name=kwargs["tool"].name,
            tool_version=kwargs["tool"].version,
            tool_input=kwargs["tool_input"],
            user_id=kwargs.get("user_id"),
        )
        self.entries[cache_key] = encode_cache_output(kwargs["output_payload"])


class _FailingCacheRepository(_FakeCacheRepository):
    def __init__(self, *, fail_get=False, fail_put=False):
        super().__init__()
        self.fail_get = fail_get
        self.fail_put = fail_put

    def get(self, **kwargs):
        self.get_calls += 1
        if self.fail_get:
            raise RuntimeError("cache lookup unavailable")
        return super().get(**kwargs)

    def put(self, **kwargs):
        self.put_calls += 1
        if self.fail_put:
            raise RuntimeError("cache store unavailable")
        super().put(**kwargs)


def test_tool_registry_serializes_calls_for_shared_rate_limit_key() -> None:
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
    registry.register(Tool(_FakeTool("news_search", make_callback("news_search")), rate_limit_key="brave", rate_limit_policy=policy))
    registry.register(Tool(_FakeTool("generic_web_search", make_callback("generic_web_search")), rate_limit_key="brave", rate_limit_policy=policy))

    thread_one = threading.Thread(target=registry.call_tool, args=("news_search", {}))
    thread_two = threading.Thread(target=registry.call_tool, args=("generic_web_search", {}))
    thread_one.start()
    thread_two.start()
    thread_one.join()
    thread_two.join()

    assert len(call_times) == 2
    assert abs(call_times[1][1] - call_times[0][1]) >= 0.045


def test_tool_registry_retries_timeout_when_policy_allows() -> None:
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
            _FakeTool("retry_tool", callback),
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
        )
    )

    result = registry.call_tool("retry_tool", {})

    assert result == {"ok": True}
    assert attempts == 2


def test_tool_registry_retries_http_429_when_policy_allows() -> None:
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
            _FakeTool("retry_tool", callback),
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
        )
    )

    result = registry.call_tool("retry_tool", {})

    assert result == {"ok": True}
    assert attempts == 2


def test_tool_registry_retries_wrapped_http_429_when_policy_allows() -> None:
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
            _FakeTool("retry_tool", callback),
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0),
        )
    )

    result = registry.call_tool("retry_tool", {})

    assert result == {"ok": True}
    assert attempts == 2


def test_tool_registry_cache_miss_invokes_tool_and_stores_result() -> None:
    cache = _FakeCacheRepository()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(_FakeTool("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2))

    result = registry.call_tool("cache_tool", {"query": "chicken"})

    assert result == {"value": 1}
    assert calls == 1
    assert cache.get_calls == 1
    assert cache.put_calls == 1


def test_tool_registry_cache_hit_skips_tool_invocation() -> None:
    cache = _FakeCacheRepository()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(_FakeTool("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2))

    first = registry.call_tool("cache_tool", {"query": "chicken"})
    second = registry.call_tool("cache_tool", {"query": "chicken"})

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert calls == 1
    assert cache.get_calls == 2
    assert cache.put_calls == 1


def test_search_meals_cache_hit_reconstructs_tool_result_with_evidence() -> None:
    cache = _FakeCacheRepository()
    calls = 0
    evidence = EvidenceView(
        item_id="meal-1",
        tool_name="search_meals",
        title="Chicken soup",
        summary="A chicken soup recipe.",
        source="search_meals",
        entity_type="meal_results",
    )

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return ToolResult(result={"name": "Chicken soup"}, evidence=[evidence])

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(_FakeTool("search_meals", callback), cache_policy=ToolCachePolicy(60), version=2))

    first = registry.call_tool("search_meals", {"query": "chicken"})
    second = registry.call_tool("search_meals", {"query": "chicken"})

    assert isinstance(first, ToolResult)
    assert isinstance(second, ToolResult)
    assert second.evidence[0].title == "Chicken soup"
    assert calls == 1


def test_tool_registry_legacy_cache_payload_is_treated_as_miss() -> None:
    cache = _FakeCacheRepository()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return ToolResult(result={"value": calls}, evidence=[])

    tool = Tool(_FakeTool("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2)
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


def test_tool_registry_malformed_cache_envelope_is_treated_as_miss() -> None:
    cache = _FakeCacheRepository()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    tool = Tool(_FakeTool("cache_tool", callback), cache_policy=ToolCachePolicy(60), version=2)
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


def test_tool_registry_cache_lookup_error_falls_back_to_tool() -> None:
    cache = _FailingCacheRepository(fail_get=True)
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(_FakeTool("cache_tool", callback), cache_policy=ToolCachePolicy(60)))

    assert registry.call_tool("cache_tool", {"query": "chicken"}) == {"value": 1}
    assert calls == 1


def test_tool_registry_cache_store_error_does_not_fail_tool_request() -> None:
    cache = _FailingCacheRepository(fail_put=True)
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(Tool(_FakeTool("cache_tool", callback), cache_policy=ToolCachePolicy(60)))

    assert registry.call_tool("cache_tool", {"query": "chicken"}) == {"value": 1}
    assert calls == 1


def test_user_specific_cache_isolated_by_user() -> None:
    cache = _FakeCacheRepository()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(
        Tool(_FakeTool("private_search", callback), cache_policy=ToolCachePolicy(60, is_user_specific=True))
    )

    with patch("tool.registry.global_registry.get_current_user_id", return_value="user-a"):
        first = registry.call_tool("private_search", {"query": "chicken"})
        second = registry.call_tool("private_search", {"query": "chicken"})
    with patch("tool.registry.global_registry.get_current_user_id", return_value="user-b"):
        third = registry.call_tool("private_search", {"query": "chicken"})

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert third == {"value": 2}
    assert calls == 2
    assert cache.put_calls == 2


def test_user_specific_cache_bypasses_anonymous_requests() -> None:
    cache = _FakeCacheRepository()
    calls = 0

    def callback(_tool_input):
        nonlocal calls
        calls += 1
        return {"value": calls}

    registry = ToolRegistry(cache_repository=cache, enable_cache=True)
    registry.register(
        Tool(_FakeTool("private_search", callback), cache_policy=ToolCachePolicy(60, is_user_specific=True))
    )

    with patch("tool.registry.global_registry.get_current_user_id", return_value=None):
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
