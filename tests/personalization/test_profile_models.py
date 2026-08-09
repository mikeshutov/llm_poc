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

from personalization.profile.models import UserAttributesSection, UserProfile
from personalization.profile.service import load_user_profile_attributes
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from request_orchestrator.constants import PLANNER_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt
from request_orchestrator.shared.prompts.render_agent_prompt import render_agent_prompt


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
    class FakeRepo:
        def list_attributes(self, *, limit=50, order_by='updated_at', descending=True, user_id=None, is_active=None, attribute_type=None, group_key=None, source=None):
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

    profile = UserProfile()

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
    management_prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction='test',
        user_profile=profile,
        include_user_attribute_management_fields=True,
    )

    default_rendered = render_agent_prompt(default_prompt)
    management_rendered = render_agent_prompt(management_prompt)

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
