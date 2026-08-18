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

from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE


def test_main_agent_profile_excludes_games_category() -> None:
    assert "games" not in MAIN_AGENT_PROFILE.allowed_category_names()
