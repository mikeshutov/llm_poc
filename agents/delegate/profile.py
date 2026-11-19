from __future__ import annotations

from pathlib import Path

from agent_runtime.models.agent_profile import (
    AgentDelegationRole,
    AgentExecutionStrategy,
    AgentName,
    AgentProfile,
)
from capabilities.discovery.discover_capabilities_tool import CAPABILITY_DISCOVERY_TOOL
from capabilities.registry.tools import TOOL_CATEGORIES, WORKSPACE_MEMORY_TOOLS
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE


WORKING_MEMORY_TEMPLATE_PATH = Path(__file__).with_name("working-memory.md")


def load_working_memory_template() -> str:
    return WORKING_MEMORY_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def build_delegate_profile() -> AgentProfile:
    return AgentProfile(
        name=AgentName.DELEGATE,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=2,
        description=(
            "Solve one task delegated by a parent agent using the available capabilities. "
            "You are provided with a structured runtime working memory as a file and return the completed result and "
            "supporting evidence to the parent. This agent cannot delegate further."
        ),
        # The delegate receives only workspace-memory capabilities and the
        # capability-discovery entry point. Domain tools must be discovered and
        # explicitly supplied by the runtime instead of being dumped here.
        allowed_categories=set(),
        excluded_categories=set(TOOL_CATEGORIES),
        static_tools=[CAPABILITY_DISCOVERY_TOOL, *WORKSPACE_MEMORY_TOOLS],
        execution_strategy=AgentExecutionStrategy.REACT,
        delegation_role=AgentDelegationRole.LEAF,
        can_delegate_agents=False,
        delegatable=True,
        max_turns=8,
        working_memory_template=load_working_memory_template(),
        planner_instruction=(
            "Solve only the task assigned by the parent agent. Use the available tools to gather, "
            "check, and organize the needed evidence. Follow the injected Working Memory instructions "
            "to maintain durable task state. Do not create a separate findings or handoff file. "
            "Return the collected evidence and a concise completion summary through the agent "
            "result. Do not answer the user directly, write the parent agent's final report, or "
            "delegate to another agent. This agent cannot delegate further."
        ),
    )


DELEGATE_PROFILE = build_delegate_profile()
