from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from common.logging import record_node_log
from agent_runtime.constants import NodeName
from agent_runtime.context import current_agent_state
from capabilities.discovery.capability_discovery import capability_discovery


class CapabilityDiscoveryArgs(BaseModel):
    prompt: str = Field(..., min_length=1, description="Describe the kind of tool or agent needed.")


def _discover(prompt: str) -> dict[str, Any]:
    state = current_agent_state.get()
    if state is None:
        return {"error": "Capability discovery is unavailable outside an active request."}
    query = prompt.strip()
    result = capability_discovery(
        state, include_agents=state.agent_profile.can_delegate_agents, query=query,
        allowed_tool_names=state.agent_profile.tool_names,
    )
    state.inputs.discovered_capabilities = result
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
