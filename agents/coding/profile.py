from __future__ import annotations

from agent_runtime.models.agent_profile import (
    AgentDelegationArgs,
    AgentExecutionStrategy,
    AgentProfile,
    AgentName,
)
from capabilities.adapters.math import calculate
from capabilities.adapters.finance import get_crypto_historical_prices
from capabilities.adapters.web import web_fetch
from capabilities.registry.constants import (
    TOOL_CATEGORY_MATH,
    TOOL_CATEGORY_FINANCE,
    TOOL_CATEGORY_FINANCE_CRYPTO,
    TOOL_CATEGORY_CODE_EXECUTION,
    TOOL_NAME_CALCULATE,
    TOOL_NAME_GET_CRYPTO_HISTORICAL_PRICES,
    TOOL_NAME_CODE_EXECUTE,
    TOOL_NAME_LOOKUP_EVIDENCE,
    TOOL_NAME_WORKSPACE_FILE_DELETE,
    TOOL_NAME_WORKSPACE_FILE_EDIT,
    TOOL_NAME_WORKSPACE_FILE_LIST,
    TOOL_NAME_WORKSPACE_FILE_READ,
    TOOL_NAME_WORKSPACE_FILE_WRITE,
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    TOOL_NAME_WEB_FETCH,
)
from capabilities.registry.tools import MEMORY_TOOLS
from capabilities.registry.tools import WORKING_MEMORY_TOOLS, WORKSPACE_TOOLS
from agent_runtime.strategies.working_memory import load_strategy_working_memory_template
from agent_runtime.planning.prompts.planner_rules import WORKING_MEMORY_LIFECYCLE_RULE
from agents.coding.prompts import build_coding_prompt
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE


_CODING_TOOL_NAMES = {
    TOOL_NAME_CALCULATE,
    TOOL_NAME_CODE_EXECUTE,
    TOOL_NAME_GET_CRYPTO_HISTORICAL_PRICES,
    TOOL_NAME_LOOKUP_EVIDENCE,
    TOOL_NAME_WORKSPACE_FILE_WRITE,
    TOOL_NAME_WORKSPACE_FILE_READ,
    TOOL_NAME_WORKSPACE_FILE_EDIT,
    TOOL_NAME_WORKSPACE_FILE_LIST,
    TOOL_NAME_WORKSPACE_FILE_DELETE,
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    TOOL_NAME_WEB_FETCH,
}


def build_coding_profile() -> AgentProfile:
    static_tools = [
        tool
        for tool in (
            calculate,
            get_crypto_historical_prices,
            web_fetch,
            *MEMORY_TOOLS,
            *WORKSPACE_TOOLS,
        )
        if tool.name in _CODING_TOOL_NAMES
    ]
    return AgentProfile(
        name=AgentName.CODING,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=8,
        description=(
            "An agent capable of generating, executing, and validating code to solve "
            "computational tasks."
        ),
        allowed_categories={
            TOOL_CATEGORY_MATH,
            TOOL_CATEGORY_FINANCE,
            TOOL_CATEGORY_FINANCE_CRYPTO,
            TOOL_CATEGORY_CODE_EXECUTION,
        },
        static_tools=[*static_tools, *WORKING_MEMORY_TOOLS],
        execution_strategy=AgentExecutionStrategy.REACT,
        delegatable=True,
        max_turns=10,
        planner_instruction=f"{WORKING_MEMORY_LIFECYCLE_RULE}\n\n{build_coding_prompt()}",
        working_memory_template=load_strategy_working_memory_template(AgentExecutionStrategy.REACT),
        delegation_args_schema=AgentDelegationArgs,
    )


CODING_PROFILE = build_coding_profile()
