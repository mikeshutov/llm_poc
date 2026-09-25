from __future__ import annotations

from collections.abc import Iterable

from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.agents.models.agent import Agent, AgentType
from request_orchestrator.agents.repository.repo_factory import get_agent_repo


SYSTEM_AGENT_PROFILES: tuple[AgentProfile, ...] = (
    MAIN_AGENT_PROFILE,
    PROFILE_MANAGEMENT_PROFILE,
)


def seed_system_agents(
    profiles: Iterable[AgentProfile] = SYSTEM_AGENT_PROFILES,
) -> list[Agent]:
    repository = get_agent_repo()
    return [
        repository.upsert(
            agent_type=AgentType.SYSTEM,
            name=profile.name,
            version=profile.version,
            description=profile.description,
            execution_strategy=profile.execution_strategy,
            allowed_categories=list(profile.allowed_categories),
            planner_instruction=profile.planner_instruction,
            planner_rules=profile.planner_rules,
            max_turns=profile.max_turns,
        )
        for profile in profiles
    ]
