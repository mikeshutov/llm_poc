from __future__ import annotations

import sys
import importlib
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from personalization.profile.models import UserProfile
from personalization.tone.models import TonePreferences
from request_orchestrator.agents.profile_management.profile import build_profile_management_profile
from request_orchestrator.shared.runtime_context import bind_runtime_context
from request_orchestrator.shared.tool_adapter.profile.set_user_display_name import set_user_display_name
from request_orchestrator.shared.tool_adapter.profile.set_user_first_name import set_user_first_name
from request_orchestrator.shared.tool_adapter.profile.set_user_last_name import set_user_last_name
from request_orchestrator.shared.tool_adapter.profile.update_user_tone import update_user_tone


def test_set_user_display_name_returns_typed_result() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                first_name="Mike",
                last_name="Shutov",
                display_name=None,
            )

        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called when profile already exists")

        def update_profile(self, **kwargs) -> UserProfile | None:
            self.profile.display_name = kwargs["display_name"]
            return self.profile

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.profile.set_user_display_name"
    )
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
            result = set_user_display_name.invoke({"display_name": "big dawg"})
    finally:
        module.get_user_profile_repo = original_repo_getter

    assert result.model_dump() == {
        "user_id": "user-123",
        "first_name": "Mike",
        "last_name": "Shutov",
        "display_name": "big dawg",
    }


def test_set_user_first_name_returns_typed_result() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                first_name=None,
                last_name="Shutov",
                display_name="big dawg",
            )

        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called when profile already exists")

        def update_profile(self, **kwargs) -> UserProfile | None:
            self.profile.first_name = kwargs["first_name"]
            return self.profile

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.profile.set_user_first_name"
    )
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
            result = set_user_first_name.invoke({"first_name": "Mike"})
    finally:
        module.get_user_profile_repo = original_repo_getter

    assert result.model_dump() == {
        "user_id": "user-123",
        "first_name": "Mike",
        "last_name": "Shutov",
        "display_name": "big dawg",
    }


def test_set_user_last_name_returns_typed_result() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                first_name="Mike",
                last_name=None,
                display_name="big dawg",
            )

        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called when profile already exists")

        def update_profile(self, **kwargs) -> UserProfile | None:
            self.profile.last_name = kwargs["last_name"]
            return self.profile

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.profile.set_user_last_name"
    )
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
            result = set_user_last_name.invoke({"last_name": "Shutov"})
    finally:
        module.get_user_profile_repo = original_repo_getter

    assert result.model_dump() == {
        "user_id": "user-123",
        "first_name": "Mike",
        "last_name": "Shutov",
        "display_name": "big dawg",
    }


def test_update_user_tone_merges_with_existing_preferences() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                tone=TonePreferences(
                    verbosity="concise",
                    humor="light",
                ),
            )
            self.updated_tone: TonePreferences | None = None

        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called when profile already exists")

        def update_profile(self, **kwargs) -> UserProfile | None:
            tone = kwargs["tone"]
            assert tone is not None
            self.updated_tone = tone
            self.profile.tone = tone
            return self.profile

    tone_tool_module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.profile.update_user_tone"
    )

    fake_repo = FakeProfileRepo()
    original_repo_getter = tone_tool_module.get_user_profile_repo
    tone_tool_module.get_user_profile_repo = lambda: fake_repo
    try:
        with bind_runtime_context(
            conversation_id="conversation-1",
            conversation_model_config=None,
            roundtrip_id="roundtrip-1",
            user_id="user-123",
        ):
            result = update_user_tone.invoke(
                {
                    "confidence": 0.95,
                    "directness": "high",
                    "technical_depth": "high",
                }
            )
    finally:
        tone_tool_module.get_user_profile_repo = original_repo_getter

    assert fake_repo.updated_tone is not None
    assert fake_repo.updated_tone.verbosity == "concise"
    assert fake_repo.updated_tone.humor == "light"
    assert fake_repo.updated_tone.directness == "high"
    assert fake_repo.updated_tone.technical_depth == "high"
    assert result.model_dump() == {
        "user_id": "user-123",
        "applied": True,
        "status": "updated",
        "reason": "",
        "confidence": 0.95,
        "minimum_confidence": 0.9,
        "tone": {
            "verbosity": "concise",
            "formality": None,
            "directness": "high",
            "humor": "light",
            "technical_depth": "high",
        },
    }


def test_update_user_tone_rejects_low_confidence_updates() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                tone=TonePreferences(
                    verbosity="concise",
                    humor="light",
                ),
            )
            self.update_called = False

        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called when profile already exists")

        def update_profile(self, **kwargs) -> UserProfile | None:
            self.update_called = True
            return self.profile

    tone_tool_module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.profile.update_user_tone"
    )

    fake_repo = FakeProfileRepo()
    original_repo_getter = tone_tool_module.get_user_profile_repo
    tone_tool_module.get_user_profile_repo = lambda: fake_repo
    try:
        with bind_runtime_context(
            conversation_id="conversation-1",
            conversation_model_config=None,
            roundtrip_id="roundtrip-1",
            user_id="user-123",
        ):
            result = update_user_tone.invoke(
                {
                    "confidence": 0.75,
                    "directness": "high",
                }
            )
    finally:
        tone_tool_module.get_user_profile_repo = original_repo_getter

    assert fake_repo.update_called is False
    assert result.model_dump() == {
        "user_id": "user-123",
        "applied": False,
        "status": "rejected",
        "reason": "Tone update rejected because confidence 0.75 is below the minimum threshold of 0.90.",
        "confidence": 0.75,
        "minimum_confidence": 0.9,
        "tone": {
            "verbosity": "concise",
            "formality": None,
            "directness": None,
            "humor": "light",
            "technical_depth": None,
        },
    }


def test_update_user_tone_skips_unchanged_updates() -> None:
    class FakeProfileRepo:
        def __init__(self) -> None:
            self.profile = UserProfile(
                user_id="user-123",
                tone=TonePreferences(
                    verbosity="concise",
                    humor="light",
                ),
            )
            self.update_called = False

        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return self.profile

        def ensure_profile(self, user_id: str) -> UserProfile:
            raise AssertionError("ensure_profile should not be called when profile already exists")

        def update_profile(self, **kwargs) -> UserProfile | None:
            self.update_called = True
            return self.profile

    tone_tool_module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.profile.update_user_tone"
    )

    fake_repo = FakeProfileRepo()
    original_repo_getter = tone_tool_module.get_user_profile_repo
    tone_tool_module.get_user_profile_repo = lambda: fake_repo
    try:
        with bind_runtime_context(
            conversation_id="conversation-1",
            conversation_model_config=None,
            roundtrip_id="roundtrip-1",
            user_id="user-123",
        ):
            result = update_user_tone.invoke(
                {
                    "confidence": 0.96,
                    "verbosity": "concise",
                    "humor": "light",
                }
            )
    finally:
        tone_tool_module.get_user_profile_repo = original_repo_getter

    assert fake_repo.update_called is False
    assert result.model_dump() == {
        "user_id": "user-123",
        "applied": False,
        "status": "unchanged",
        "reason": "Tone update skipped because the requested values do not change the stored tone preference.",
        "confidence": 0.96,
        "minimum_confidence": 0.9,
        "tone": {
            "verbosity": "concise",
            "formality": None,
            "directness": None,
            "humor": "light",
            "technical_depth": None,
        },
    }


def test_profile_management_profile_includes_update_user_tone_tool() -> None:
    profile = build_profile_management_profile(UserProfile())
    tool_names = [tool.name for tool in profile.extra_tools]

    assert "update_user_tone" in tool_names
