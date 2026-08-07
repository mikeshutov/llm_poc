from __future__ import annotations

from request_orchestrator.agents.models.agent_profile import AgentProfile
from request_orchestrator.shared.tool_adapter.memories.search_memories import search_memories
from request_orchestrator.shared.tool_adapter.memories.search_roundtrip_memories import search_roundtrip_memories
from request_orchestrator.shared.tool_adapter.user_attributes.get_user_attributes import get_user_attributes
from request_orchestrator.shared.tool_adapter.user_attributes.search_user_attributes import search_user_attributes
from tool.tools import TOOL_CATEGORIES

READ_ONLY_PROFILE_AND_MEMORY_TOOLS = [
    search_memories,
    search_roundtrip_memories,
]

MAIN_AGENT_PROFILE = AgentProfile(
    name='main_agent',
    allowed_categories=set(TOOL_CATEGORIES.keys()) - {'memories', 'user_attributes'},
    extra_tools=READ_ONLY_PROFILE_AND_MEMORY_TOOLS,
)
