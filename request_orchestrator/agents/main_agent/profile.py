from __future__ import annotations

from request_orchestrator.agents.models.agent_profile import AgentProfile
from tool.tools import TOOL_CATEGORIES


MAIN_AGENT_PROFILE = AgentProfile(
    name='main_agent',
    allowed_categories=set(TOOL_CATEGORIES.keys()),
    extra_tools=[],
)
