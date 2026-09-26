from uuid import uuid4

import pytest

from request_orchestrator.agents.models.agent import Agent, AgentType
from request_orchestrator.agents import seeder


def _agent(**overrides) -> Agent:
    payload = {
        "id": uuid4(),
        "name": "researcher",
        "planner_instruction": "Plan the request.",
        **overrides,
    }
    return Agent.model_validate(payload)


def test_user_agent_requires_user_id() -> None:
    with pytest.raises(ValueError, match="require a user_id"):
        _agent(agent_type=AgentType.USER)


def test_system_agent_cannot_have_user_id() -> None:
    with pytest.raises(ValueError, match="cannot have a user_id"):
        _agent(agent_type=AgentType.SYSTEM, user_id="user-1")


def test_system_agent_can_be_global() -> None:
    agent = _agent(agent_type=AgentType.SYSTEM)

    assert agent.user_id is None
    assert agent.agent_type == AgentType.SYSTEM


def test_system_agent_seeder_delegates_each_profile(monkeypatch) -> None:
    profiles = [seeder.MAIN_AGENT_PROFILE]
    seeded = [object()]
    calls = []

    class FakeRepository:
        def list_system_agents(self):
            return []

        def upsert(self, **kwargs):
            calls.append(kwargs)
            return seeded[0]

    monkeypatch.setattr(seeder, "get_agent_repo", lambda: FakeRepository())

    assert seeder.seed_system_agents(profiles) == seeded
    assert calls == [{
        "agent_type": AgentType.SYSTEM,
        "name": profiles[0].name,
        "version": profiles[0].version,
        "description": profiles[0].description,
        "execution_strategy": profiles[0].execution_strategy,
        "allowed_categories": list(profiles[0].allowed_categories),
        "planner_instruction": profiles[0].planner_instruction,
        "planner_rules": profiles[0].planner_rules,
        "max_turns": profiles[0].max_turns,
    }]


def test_system_agent_seeder_deletes_profiles_removed_from_code(monkeypatch) -> None:
    profiles = [seeder.MAIN_AGENT_PROFILE]
    stale_agent = _agent(agent_type=AgentType.SYSTEM, name="removed-agent")
    deleted = []

    class FakeRepository:
        def list_system_agents(self):
            return [stale_agent]

        def delete_system_agents(self, names):
            deleted.append(names)

        def upsert(self, **kwargs):
            return kwargs

    monkeypatch.setattr(seeder, "get_agent_repo", lambda: FakeRepository())

    seeder.seed_system_agents(profiles)

    assert deleted == [{"removed-agent"}]
