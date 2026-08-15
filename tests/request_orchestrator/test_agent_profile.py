from __future__ import annotations

import sys
from types import SimpleNamespace
from types import ModuleType

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules["pycountry"] = pycountry_module

from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from tool.tools import TOOL_CATEGORIES


def test_agent_profile_caches_resolved_tools_from_constructor_inputs() -> None:
    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        allowed_categories={"user_attributes"},
        extra_tools=[
            SimpleNamespace(name="custom_tool"),
        ],
    )

    expected_category_tool_names = {
        tool.name
        for tool in TOOL_CATEGORIES["user_attributes"].tools
    }

    assert expected_category_tool_names.issubset(profile.tool_names)
    assert "custom_tool" in profile.tool_names
    assert {tool.name for tool in profile.tools} == profile.tool_names


def test_agent_profile_extra_tools_override_category_tools_by_name() -> None:
    category_tool = TOOL_CATEGORIES["user_attributes"].tools[0]
    replacement_tool = SimpleNamespace(name=category_tool.name, replacement=True)

    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        allowed_categories={"user_attributes"},
        extra_tools=[replacement_tool],
    )

    resolved_tool = next(tool for tool in profile.tools if tool.name == category_tool.name)
    assert resolved_tool is replacement_tool
