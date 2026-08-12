from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from dataclasses import dataclass
from uuid import uuid4


if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from personalization.profile.models import GeoLocation, GeoMetadata, UserAttributesSection, UserProfile
from personalization.profile.service import build_user_profile, hydrate_user_profile_core, load_user_profile_attributes
from personalization.tone.models import TonePreferences
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from request_orchestrator.constants import PLANNER_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, PromptSectionKeys
from request_orchestrator.models.synthesized_result import DEFAULT_SYNTHESIS_FOLLOW_UP, SynthesisResult
from common.data import repair_common_json_issues


@dataclass
class LegacyUserAttribute:
    id: object
    user_id: str | None
    value: list[str]
    attribute_embedding: list[float] | None
    attribute_type: str
    group_key: str | None
    source: str | None
    is_active: bool
    created_at: str
    updated_at: str
    confidence: float | None
    importance: float | None


def test_user_attributes_section_accepts_dataclass_shaped_attributes() -> None:
    legacy_attribute = LegacyUserAttribute(
        id=uuid4(),
        user_id=None,
        value=["pizza", "eggs"],
        attribute_embedding=[0.1, 0.2],
        attribute_type="food.likes",
        group_key=None,
        source="explicit",
        is_active=True,
        created_at="2026-08-05T00:00:00Z",
        updated_at="2026-08-05T00:00:00Z",
        confidence=0.9,
        importance=0.8,
    )

    section = UserAttributesSection(attributes=[legacy_attribute])

    assert len(section.attributes) == 1
    assert section.attributes[0].attribute_type == "food.likes"
    assert section.attributes[0].value == ["pizza", "eggs"]


def test_load_user_profile_attributes_condenses_same_type_and_group_key() -> None:
    captured_user_ids: list[str | None] = []

    class FakeRepo:
        def list_attributes(self, *, limit=50, order_by='updated_at', descending=True, user_id=None, is_active=None, attribute_type=None, group_key=None, source=None):
            captured_user_ids.append(user_id)
            return [
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=['pizza', 'eggs'],
                    attribute_embedding=None,
                    attribute_type='food.likes',
                    group_key=None,
                    source='explicit',
                    is_active=True,
                    created_at='2026-08-05T00:00:00Z',
                    updated_at='2026-08-05T00:00:00Z',
                    confidence=0.9,
                    importance=0.8,
                ),
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=['eggs', 'coffee'],
                    attribute_embedding=None,
                    attribute_type='food.likes',
                    group_key=None,
                    source='explicit',
                    is_active=True,
                    created_at='2026-08-04T00:00:00Z',
                    updated_at='2026-08-04T00:00:00Z',
                    confidence=0.8,
                    importance=0.7,
                ),
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=['cake'],
                    attribute_embedding=None,
                    attribute_type='food.likes',
                    group_key='desserts',
                    source='explicit',
                    is_active=True,
                    created_at='2026-08-03T00:00:00Z',
                    updated_at='2026-08-03T00:00:00Z',
                    confidence=0.8,
                    importance=0.7,
                ),
            ]

    profile = UserProfile(user_id='user-123')

    import personalization.profile.service as profile_service
    original_repo_getter = profile_service.get_user_attribute_repo
    profile_service.get_user_attribute_repo = lambda: FakeRepo()
    try:
        load_user_profile_attributes(
            profile,
            requested_attribute_types=['food.likes'],
        )
    finally:
        profile_service.get_user_attribute_repo = original_repo_getter

    assert len(profile.user_attributes.attributes) == 2

    ungrouped = next(attribute for attribute in profile.user_attributes.attributes if attribute.group_key is None)
    desserts = next(attribute for attribute in profile.user_attributes.attributes if attribute.group_key == 'desserts')

    assert ungrouped.attribute_type == 'food.likes'
    assert ungrouped.value == ['pizza', 'eggs', 'coffee']
    assert desserts.value == ['cake']
    assert captured_user_ids == ['user-123']


def test_prompt_profile_excludes_management_fields_by_default() -> None:
    profile = UserProfile(
        user_attributes=UserAttributesSection(
            attributes=[
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=['pizza'],
                    attribute_embedding=None,
                    attribute_type='food.likes',
                    group_key='meals',
                    source='explicit',
                    is_active=True,
                    created_at='2026-08-05T00:00:00Z',
                    updated_at='2026-08-05T00:00:00Z',
                    confidence=0.9,
                    importance=0.8,
                )
            ]
        )
    )

    default_prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction='test',
        user_profile=profile,
    )
    default_prompt.include_user_profile()
    management_prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction='test',
        user_profile=profile,
    )
    management_prompt.include_user_profile(include_management_fields=True)

    default_rendered = default_prompt.prompt_text()
    management_rendered = management_prompt.prompt_text()

    assert 'food.likes' in default_rendered
    assert 'pizza' in default_rendered
    assert 'group_key' not in default_rendered
    assert 'source' not in default_rendered
    assert 'created_at' not in default_rendered
    assert 'updated_at' not in default_rendered
    assert 'is_active' not in default_rendered
    assert '"id"' not in default_rendered

    assert 'group_key' in management_rendered
    assert 'source' in management_rendered
    assert 'created_at' in management_rendered
    assert 'updated_at' in management_rendered
    assert 'is_active' in management_rendered
    assert '"id"' in management_rendered



def test_prompt_profile_prunes_empty_fields() -> None:
    profile = UserProfile()
    rendered = profile.to_prompt_dict()

    assert rendered == {}


def test_prompt_profile_includes_shared_tone_preferences() -> None:
    profile = UserProfile(
        tone=TonePreferences(
            verbosity="concise",
            formality="casual",
            directness="high",
            humor="light",
            technical_depth="high",
        )
    )

    rendered = profile.to_prompt_dict(include_tone=True)

    assert rendered["tone"] == {
        "verbosity": "concise",
        "formality": "casual",
        "directness": "high",
        "humor": "light",
        "technical_depth": "high",
    }


def test_prompt_profile_excludes_geometadata_latitude_and_longitude() -> None:
    profile = UserProfile(
        geometadata=GeoMetadata(
            current_datetime="2026-08-11T09:07:57.790737-04:00",
            current_weekday="Tuesday",
            timezone="America/Toronto",
            location=GeoLocation(
                city="Toronto",
                region="Ontario",
                country="Canada",
                latitude=43.6576,
                longitude=-79.3798,
                timezone="America/Toronto",
            ),
        )
    )

    rendered = profile.to_prompt_dict()

    assert rendered["geometadata"] == {
        "current_datetime": "2026-08-11T09:07:57.790737-04:00",
        "current_weekday": "Tuesday",
        "timezone": "America/Toronto",
        "location": {
            "city": "Toronto",
            "region": "Ontario",
            "country": "Canada",
        },
    }


def test_prompt_profile_excludes_tone_by_default() -> None:
    profile = UserProfile(
        tone=TonePreferences(
            verbosity="concise",
            formality="casual",
        )
    )

    rendered = profile.to_prompt_dict()

    assert "tone" not in rendered


def test_user_profile_model_dump_includes_stored_tone() -> None:
    profile = UserProfile(
        user_id='user-123',
        tone=TonePreferences(
            verbosity='concise',
            directness='high',
            humor='light',
        ),
    )

    dumped = profile.model_dump()

    assert dumped["tone"] == {
        "verbosity": "concise",
        "formality": None,
        "directness": "high",
        "humor": "light",
        "technical_depth": None,
    }


def test_repair_common_json_issues_replaces_semicolon_between_fields() -> None:
    raw = '{"result":["a"]; "clarifying_question":"","follow_up":""}'

    repaired = repair_common_json_issues(raw)

    assert repaired == '{"result":["a"], "clarifying_question":"","follow_up":""}'


def test_synthesis_result_requires_exactly_one_question_field() -> None:
    result = SynthesisResult.model_validate(
        {
            "result": ["done"],
            "follow_up": "Do you want more detail?",
            "clarifying_question": "",
            "roundtrip_summary": "summary",
            "tool_summary": {},
        }
    )

    assert result.follow_up == "Do you want more detail?"
    assert result.clarifying_question == ""


def test_synthesis_result_falls_back_when_both_question_fields_are_empty() -> None:
    result = SynthesisResult.model_validate(
        {
            "result": ["done"],
            "follow_up": "",
            "clarifying_question": "",
            "roundtrip_summary": "summary",
            "tool_summary": {},
        }
    )

    assert result.follow_up == DEFAULT_SYNTHESIS_FOLLOW_UP
    assert result.clarifying_question == ""


def test_synthesis_result_prefers_clarifying_question_when_both_question_fields_are_set() -> None:
    result = SynthesisResult.model_validate(
        {
            "result": ["done"],
            "follow_up": "Do you want more detail?",
            "clarifying_question": "Which option do you mean?",
            "roundtrip_summary": "summary",
            "tool_summary": {},
        }
    )

    assert result.follow_up == ""
    assert result.clarifying_question == "Which option do you mean?"


def test_hydrate_user_profile_core_loads_persisted_tone() -> None:
    persisted_profile = UserProfile(
        user_id="user-123",
        tone=TonePreferences(
            verbosity="concise",
            technical_depth="high",
        ),
    )

    class FakeProfileRepo:
        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return persisted_profile

    import personalization.profile.service as profile_service
    original_repo_getter = profile_service.get_user_profile_repo
    profile_service.get_user_profile_repo = lambda: FakeProfileRepo()
    try:
        hydrated = hydrate_user_profile_core(UserProfile(user_id="user-123"))
    finally:
        profile_service.get_user_profile_repo = original_repo_getter

    assert hydrated.tone is not None
    assert hydrated.tone.verbosity == "concise"
    assert hydrated.tone.technical_depth == "high"


def test_build_user_profile_preserves_input_tone_without_persisted_profile() -> None:
    class FakeProfileRepo:
        def get_profile(self, user_id: str) -> UserProfile | None:
            assert user_id == "user-123"
            return None

    import personalization.profile.service as profile_service
    original_repo_getter = profile_service.get_user_profile_repo
    profile_service.get_user_profile_repo = lambda: FakeProfileRepo()
    try:
        profile = build_user_profile(
            user_id="user-123",
            tone=TonePreferences(
                formality="casual",
                directness="high",
            ),
        )
    finally:
        profile_service.get_user_profile_repo = original_repo_getter

    assert profile.tone is not None
    assert profile.tone.formality == "casual"
    assert profile.tone.directness == "high"


def test_agent_prompt_exposes_prompt_text_sections_and_token_count() -> None:
    prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction='test',
        user_profile=UserProfile(user_id='user-123'),
        task='Find boots.',
    )
    prompt.include_user_profile().include_latest_user_prompt().include_task(heading='Goal:')

    prompt_text = prompt.prompt_text()
    prompt_token_count = prompt.prompt_token_count()

    assert prompt_text
    assert prompt_token_count > 0
    assert [section.heading for section in prompt.included_sections()] == [
        'User Profile (JSON):',
        'Latest User Prompt:',
        'Goal:',
    ]
    assert [section.key for section in prompt.included_sections()] == [
        PromptSectionKeys.USER_PROFILE,
        PromptSectionKeys.LATEST_USER_PROMPT,
        PromptSectionKeys.TASK,
    ]
    assert prompt.get_section(PromptSectionKeys.TASK) is not None
    assert prompt.get_section(PromptSectionKeys.TASK).value.text == 'Find boots.'
    assert prompt.get_section(PromptSectionKeys.TASK).value.token_count > 0
    assert prompt.get_section_content(PromptSectionKeys.TASK) == 'Find boots.'


def test_agent_prompt_user_profile_includes_tone_only_when_requested() -> None:
    profile = UserProfile(
        user_id='user-123',
        tone=TonePreferences(
            verbosity='concise',
            directness='high',
        ),
    )

    default_prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction='test',
        user_profile=profile,
    )
    default_prompt.include_user_profile()

    tone_prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction='test',
        user_profile=profile,
    )
    tone_prompt.include_user_profile(include_tone=True)

    assert '"tone"' not in default_prompt.prompt_text()
    assert '"verbosity": "concise"' not in default_prompt.prompt_text()

    assert '"tone"' in tone_prompt.prompt_text()
    assert '"verbosity": "concise"' in tone_prompt.prompt_text()
    assert '"directness": "high"' in tone_prompt.prompt_text()


def test_agent_prompt_to_dict_includes_tone_when_user_profile_section_requested_with_tone() -> None:
    profile = UserProfile(
        user_id='user-123',
        tone=TonePreferences(
            verbosity='concise',
            directness='high',
        ),
    )

    prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction='test',
        user_profile=profile,
    )
    prompt.include_user_profile(include_tone=True)

    payload = prompt.to_dict()

    assert payload["user_profile"]["tone"] == {
        "verbosity": "concise",
        "directness": "high",
    }



