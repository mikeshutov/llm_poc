from __future__ import annotations

import importlib
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

from agent_runtime.context import bind_runtime_context
from agents.profile_management.profile import build_profile_management_profile
from agents.profile_management.prompts import build_profile_management_rules
from capabilities.adapters.profile.update_user_names import update_user_names
from capabilities.adapters.profile.update_user_tone import update_user_tone
from personalization.profile.models import UserProfile
from personalization.tone.models import TonePreferences
from retrieval.evidence import ToolResult


def test_update_user_names_updates_supplied_fields_and_ignores_empty_fields() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                first_name="Mike",
                last_name="Shutov",
                display_name="Mike",
            )
            self.update_calls = 0

        def get_profile(self, user_id: str) -> UserProfile | None:
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called")

        def update_profile(self, **kwargs) -> UserProfile | None:
            self.update_calls += 1
            self.profile.first_name = kwargs["first_name"]
            self.profile.last_name = kwargs["last_name"]
            self.profile.display_name = kwargs["display_name"]
            return self.profile

    module = importlib.import_module("capabilities.adapters.profile.update_user_names")
    fake_repo = FakeProfileRepo()
    original_repo_getter = module.get_user_profile_repo
    module.get_user_profile_repo = lambda: fake_repo
    try:
        with bind_runtime_context(
            conversation_id="conversation-1",
            conversation_model_config=None,
            roundtrip_id="roundtrip-1",
            user_id="user-123",
        ):
            result = update_user_names.invoke(
                {"first_name": "Michael", "last_name": "", "display_name": ""}
            )
    finally:
        module.get_user_profile_repo = original_repo_getter

    assert isinstance(result, ToolResult)
    assert result.result.first_name == "Michael"
    assert result.result.last_name == "Shutov"
    assert result.result.display_name == "Mike"
    assert fake_repo.update_calls == 1


def test_update_user_tone_merges_with_existing_preferences() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                tone=TonePreferences(verbosity="concise", humor="light"),
            )
            self.updated_tone = None

        def get_profile(self, user_id: str) -> UserProfile | None:
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called")

        def update_profile(self, **kwargs) -> UserProfile | None:
            self.updated_tone = kwargs["tone"]
            self.profile.tone = self.updated_tone
            return self.profile

    module = importlib.import_module("capabilities.adapters.profile.update_user_tone")
    fake_repo = FakeProfileRepo()
    original_repo_getter = module.get_user_profile_repo
    module.get_user_profile_repo = lambda: fake_repo
    try:
        with bind_runtime_context(
            conversation_id="conversation-1",
            conversation_model_config=None,
            roundtrip_id="roundtrip-1",
            user_id="user-123",
        ):
            result = update_user_tone.invoke(
                {"confidence": 0.95, "directness": "high", "technical_depth": "high"}
            )
    finally:
        module.get_user_profile_repo = original_repo_getter

    assert isinstance(result, ToolResult)
    assert result.result.applied is True
    assert result.result.tone.directness == "high"


def test_profile_management_profile_includes_combined_name_tool() -> None:
    profile = build_profile_management_profile(UserProfile())
    tool_names = [tool.name for tool in profile.static_tools]

    assert "user_update_names" in tool_names
    assert "user_set_display_name" not in tool_names
    assert "user_set_first_name" not in tool_names
    assert "user_set_last_name" not in tool_names


def test_profile_management_rules_use_supported_tone_values() -> None:
    rules = build_profile_management_rules()

    assert "closest available supported tone/formality value" in rules
    assert "do not invent new tone values" in rules
    assert "clearly asks for that preference to be remembered" in rules
    assert "completed update" in rules
