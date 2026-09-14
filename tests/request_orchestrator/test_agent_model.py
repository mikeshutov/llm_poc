from uuid import uuid4

import pytest

from request_orchestrator.agents.models.agent import Agent, AgentType


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
