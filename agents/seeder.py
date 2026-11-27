from __future__ import annotations

from collections.abc import Iterable

from agent_runtime.models.agent_profile import AgentProfile
from capabilities.registry.global_registry import REGISTERED_AGENTS
from agents.models.agent import Agent, AgentType
from agents.repository.repo_factory import get_agent_repo


def seed_system_agents(
    profiles: Iterable[AgentProfile] = REGISTERED_AGENTS,
) -> list[Agent]:
    repository = get_agent_repo()
    profiles = tuple(profiles)
    profile_names = {profile.name for profile in profiles}
    existing_names = {agent.name for agent in repository.list_system_agents()}
    stale_names = existing_names - profile_names
    if stale_names:
        repository.delete_system_agents(stale_names)

    return [
        repository.upsert(
            agent_type=AgentType.SYSTEM,
            name=profile.name,
            version=profile.version,
            description=profile.description,
            execution_strategy=profile.execution_strategy,
            allowed_categories=list(profile.allowed_categories),
            planner_instruction=profile.planner_instruction,
            max_turns=profile.max_turns,
        )
        for profile in profiles
    ]
