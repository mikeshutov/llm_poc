from __future__ import annotations


from agents.main.request import TOP_LEVEL_PROFILE


def test_top_level_profile_is_owned_by_main_request_strategy() -> None:
    assert TOP_LEVEL_PROFILE.name == "request_orchestrator"


def test_top_level_profile_exposes_static_capabilities() -> None:
    assert {tool.name for tool in TOP_LEVEL_PROFILE.static_tools} == {
        "discover_capabilities",
        "math_calculate",
        "user_management",
        "update_memory_state",
    }
