from __future__ import annotations

from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentName,
    AgentProfile,
)
from capabilities.discovery.discover_capabilities_tool import CAPABILITY_DISCOVERY_TOOL
from capabilities.registry.constants import (
    TOOL_CATEGORY_PRODUCTS,
    TOOL_CATEGORY_WEATHER,
    TOOL_CATEGORY_FINANCE,
    TOOL_CATEGORY_FINANCE_CRYPTO,
    TOOL_CATEGORY_WEB_SEARCH,
    TOOL_CATEGORY_KNOWLEDGE,
    TOOL_CATEGORY_CALENDAR,
    TOOL_CATEGORY_LOCATION,
    TOOL_CATEGORY_BOOKS,
    TOOL_CATEGORY_LANGUAGE,
    TOOL_CATEGORY_FOOD,
    TOOL_CATEGORY_MATH,
    TOOL_CATEGORY_CODE_EXECUTION,
    TOOL_CATEGORY_GAMES,
    TOOL_CATEGORY_MEMORIES,
    TOOL_CATEGORY_FILES,
    TOOL_CATEGORY_WORKSPACE,
)
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE


def build_delegate_profile() -> AgentProfile:
    return AgentProfile(
        name=AgentName.DELEGATE,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=9,
        description=(
            "Solve one task delegated by a parent agent using the available capabilities. "
            "Gather supporting evidence for the parent. This agent cannot delegate further."
        ),
        # The profile grants domain permissions; discovery selects which of
        # those tools are exposed for the assigned task.
        allowed_categories={
            TOOL_CATEGORY_PRODUCTS,
            TOOL_CATEGORY_WEATHER,
            TOOL_CATEGORY_FINANCE,
            TOOL_CATEGORY_FINANCE_CRYPTO,
            TOOL_CATEGORY_WEB_SEARCH,
            TOOL_CATEGORY_KNOWLEDGE,
            TOOL_CATEGORY_CALENDAR,
            TOOL_CATEGORY_LOCATION,
            TOOL_CATEGORY_BOOKS,
            TOOL_CATEGORY_LANGUAGE,
            TOOL_CATEGORY_FOOD,
            TOOL_CATEGORY_MATH,
            TOOL_CATEGORY_CODE_EXECUTION,
            TOOL_CATEGORY_GAMES,
            TOOL_CATEGORY_MEMORIES,
            TOOL_CATEGORY_FILES,
            TOOL_CATEGORY_WORKSPACE,
        },
        static_tools=[CAPABILITY_DISCOVERY_TOOL],
        execution_strategy=AgentExecutionStrategy.REACT,
        can_delegate_agents=False,
        delegatable=True,
        max_turns=10,
        planner_instruction=(
            "Complete the assigned task using available tools. Stay within its scope and "
            "ground findings in supporting evidence."
        ),
    )


DELEGATE_PROFILE = build_delegate_profile()
