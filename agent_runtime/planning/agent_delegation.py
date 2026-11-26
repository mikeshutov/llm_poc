from __future__ import annotations

from collections.abc import Iterable

from langchain_core.tools import BaseTool, StructuredTool

from agent_runtime.models.agent_state import AgentState


def delegated_agent_states(state: AgentState) -> list[AgentState]:
    """Return the agent states visible to the current agent."""
    if not state.agent_profile.can_delegate_agents:
        return []

    capabilities = state.inputs.discovered_capabilities
    discovered_names = {
        agent.agent
        for agent in ([] if capabilities is None else capabilities.agents)
    }
    defaults = state.agent_profile.default_available_agent_names
    return [
        delegated
        for delegated in (state.available_agent_states or {}).values()
        if delegated.agent_profile.name != state.agent_profile.name
        and delegated.agent_profile.delegatable
        and (
            delegated.agent_profile.name in defaults
            or capabilities is None
            or delegated.agent_profile.name in discovered_names
        )
    ]


def build_agent_delegation_tools(
    agent_states: Iterable[AgentState],
) -> list[BaseTool]:
    """Build native tools that create explicit agent-delegation plan steps."""
    tools: list[BaseTool] = []
    for agent_state in agent_states:
        profile = agent_state.agent_profile
        args_schema = profile.delegation_args_schema

        def delegate(*, agent_name=profile.name, **kwargs):
            payload = dict(kwargs)
            evidence_ids = payload.get("evidence_ids")
            if isinstance(evidence_ids, list):
                payload["evidence_ids"] = list(dict.fromkeys(
                    value.strip()
                    for value in evidence_ids
                    if isinstance(value, str) and value.strip()
                ))
            return {"agent": agent_name, **payload}

        tools.append(
            StructuredTool.from_function(
                func=delegate,
                name=profile.name,
                description=profile.description,
                args_schema=args_schema,
                metadata={"agent_delegation": True},
            )
        )
    return tools
