from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from common.logging import record_node_log
from agent_runtime.constants import NodeName
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.main_state import MainState
from capabilities.discovery.capability_discovery import capability_discovery


class CapabilityDiscoveryArgs(BaseModel):
    prompt: str = Field(..., min_length=1, description="Describe the kind of tool or agent needed.")


_CURRENT_STATE: ContextVar[MainState | None] = ContextVar("capability_discovery_state", default=None)
_CURRENT_AGENT_STATE: ContextVar[AgentState | None] = ContextVar("capability_discovery_agent_state", default=None)


def bind_capability_discovery_state(state: MainState):
    return _CURRENT_STATE.set(state)


def reset_capability_discovery_state(token) -> None:
    _CURRENT_STATE.reset(token)


def bind_capability_discovery_agent_state(state: AgentState):
    return _CURRENT_AGENT_STATE.set(state)


def reset_capability_discovery_agent_state(token) -> None:
    _CURRENT_AGENT_STATE.reset(token)


def _discover(prompt: str) -> dict[str, Any]:
    state = _CURRENT_STATE.get()
    if state is None:
        return {"error": "Capability discovery is unavailable outside an active request."}
    query = prompt.strip()
    state.discovery_query = query
    state.discovery_query_embedding = None
    result = capability_discovery(state, include_agents=True, query=query)
    state.discovered_capabilities = result
    agent_state = _CURRENT_AGENT_STATE.get()
    if agent_state is not None:
        agent_state.inputs.discovered_capabilities = result
    payload = {
        "prompt": query,
        "tools": [tool.model_dump(mode="json") for tool in result.tools],
        "agents": [agent.model_dump(mode="json") for agent in result.agents],
    }
    record_node_log(
        roundtrip_id=state.execution_context.roundtrip_id,
        node_name=NodeName.CAPABILITY_DISCOVERY,
        stage="runtime_tool",
        event_type="capability_discovery_tool",
        message="Runtime capability discovery completed",
        payload={
            "prompt": query,
            "tool_count": len(result.tools),
            "agent_count": len(result.agents),
        },
    )
    return payload


CAPABILITY_DISCOVERY_TOOL = StructuredTool.from_function(
    func=_discover,
    name="discover_capabilities",
    description=(
        "Find tools or agents for a capability that is not currently available. "
        "Describe the needed capability in the prompt argument."
    ),
    args_schema=CapabilityDiscoveryArgs,
)
