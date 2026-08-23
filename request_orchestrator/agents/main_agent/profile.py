from __future__ import annotations

from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.shared.tool_adapter.memories.get_memory_detail import get_memory_detail
from request_orchestrator.shared.tool_adapter.memories.search_memories import search_memories
from request_orchestrator.shared.tool_adapter.memories.search_roundtrip_memories import search_roundtrip_memories
from tool.tools import TOOL_CATEGORIES

READ_ONLY_PROFILE_AND_MEMORY_TOOLS = [
    get_memory_detail,
    search_memories,
    search_roundtrip_memories,
]

MAIN_AGENT_PROFILE = AgentProfile(
    name='main_agent',
    scope=MAIN_AGENT_MODEL_SCOPE,
    allowed_categories=set(TOOL_CATEGORIES.keys()) - {'games', 'memories', 'user_attributes'},
    extra_tools=READ_ONLY_PROFILE_AND_MEMORY_TOOLS,
)
