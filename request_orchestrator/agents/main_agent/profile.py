from __future__ import annotations

from conversation.models.conversation_model_config import (
    DEFAULT_MAIN_AGENT_PLANNER_MODEL,
    DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL,
    DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL,
    MAIN_AGENT_MODEL_SCOPE,
)
from request_orchestrator.models.agent_profile import AgentProfile
from request_orchestrator.shared.tool_adapter.memories.get_memory_detail import get_memory_detail
from request_orchestrator.shared.tool_adapter.memories.search_memories import search_memories
from request_orchestrator.shared.tool_adapter.memories.search_roundtrip_memories import search_roundtrip_memories
from request_orchestrator.shared.tool_adapter.user_attributes.get_user_attributes import get_user_attributes
from request_orchestrator.shared.tool_adapter.user_attributes.search_user_attributes import search_user_attributes
from tool.tools import TOOL_CATEGORIES

READ_ONLY_PROFILE_AND_MEMORY_TOOLS = [
    get_memory_detail,
    search_memories,
    search_roundtrip_memories,
]

MAIN_AGENT_PROFILE = AgentProfile(
    name='main_agent',
    scope=MAIN_AGENT_MODEL_SCOPE,
    allowed_categories=set(TOOL_CATEGORIES.keys()) - {'memories', 'user_attributes'},
    extra_tools=READ_ONLY_PROFILE_AND_MEMORY_TOOLS,
    default_stage_models={
        "request_analysis": DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL,
        "planner": DEFAULT_MAIN_AGENT_PLANNER_MODEL,
        "synthesis": DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL,
    },
)
