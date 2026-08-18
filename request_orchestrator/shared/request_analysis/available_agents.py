from __future__ import annotations

from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.request_analysis.models import AvailableAgent, AvailableAgentToolCategory


def build_available_agents(main_state: MainState) -> list[AvailableAgent]:
    available_agents: list[AvailableAgent] = []
    for agent_state in main_state.agent_states.values():
        available_agents.append(
            AvailableAgent(
                agent=agent_state.agent_profile.name,
                description=agent_state.agent_profile.description,
                tool_categories=[
                    AvailableAgentToolCategory(
                        name=name,
                        description=category.description,
                    )
                    for name, category in sorted(agent_state.agent_profile.tool_categories.items())
                ],
            )
        )
    return available_agents
