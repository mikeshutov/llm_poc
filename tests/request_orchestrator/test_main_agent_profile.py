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

from request_orchestrator.strategies.main_request_strategy import TOP_LEVEL_PROFILE


def test_top_level_profile_is_owned_by_main_request_strategy() -> None:
    assert TOP_LEVEL_PROFILE.name == "request_orchestrator"
