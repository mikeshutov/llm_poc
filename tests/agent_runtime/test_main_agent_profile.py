from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(
        lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper())
    )
    sys.modules["pycountry"] = pycountry_module

from agents.main.request import TOP_LEVEL_PROFILE


def test_top_level_profile_is_owned_by_main_request_strategy() -> None:
    assert TOP_LEVEL_PROFILE.name == "request_orchestrator"


def test_top_level_profile_exposes_static_capabilities() -> None:
    assert {tool.name for tool in TOP_LEVEL_PROFILE.static_tools} == {
        "discover_capabilities",
        "user_management",
    }
