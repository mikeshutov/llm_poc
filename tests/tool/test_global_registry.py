from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from common.http import HttpClientError
from requests.exceptions import Timeout
from tool.registry.global_registry import ToolRegistry
from tool.models import RateLimitPolicy, RetryPolicy, Tool


class _FakeTool:
    def __init__(self, name: str, callback) -> None:
        self.name = name
        self.description = ""
        self.args_schema = None
        self._callback = callback

    def invoke(self, tool_input=None):
        return self._callback(tool_input or {})


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
