from __future__ import annotations

from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentProfile,
    AgentName,
)
from agents.research.prompts import build_research_rules
from agent_runtime.strategies.working_memory import load_strategy_working_memory_template
from agent_runtime.planning.prompts.planner_rules import WORKING_MEMORY_LIFECYCLE_RULE
from capabilities.registry.tools import WORKING_MEMORY_TOOLS
from capabilities.registry.constants import (
    TOOL_CATEGORY_WEB_SEARCH,
    TOOL_CATEGORY_KNOWLEDGE,
    TOOL_CATEGORY_BOOKS,
    TOOL_CATEGORY_LANGUAGE,
)
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE


def build_research_profile() -> AgentProfile:
    return AgentProfile(
        name=AgentName.RESEARCH,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=12,
        description=(
            "Research a topic by decomposing it into questions, selecting appropriate "
            "sources, gathering and checking evidence, and recording structured findings in "
            "runtime working memory for a downstream document-writing agent."
        ),
        allowed_categories={
            TOOL_CATEGORY_WEB_SEARCH,
            TOOL_CATEGORY_KNOWLEDGE,
            TOOL_CATEGORY_BOOKS,
            TOOL_CATEGORY_LANGUAGE,
        },
        static_tools=list(WORKING_MEMORY_TOOLS),
        default_available_agent_names={AgentName.DELEGATE},
        execution_strategy=AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR,
        delegatable=True,
        max_turns=10,
        planner_instruction=(
            f"{WORKING_MEMORY_LIFECYCLE_RULE}\n\n"
            "Plan and execute a rigorous research workflow for the provided topic. "
            "Delegate independent research questions when useful. Give each delegate a focused task, "
            "reconcile its findings, and avoid duplicating completed work. "
            "Use working memory as the complete research handoff: organize it so a downstream "
            "document-writing agent can produce the requested file without repeating the research. "
            "Include the research scope and methodology, key findings, source URLs and dates, "
            "evidence references, disagreements, uncertainty, conclusions, and remaining gaps.\n\n"
            f"{build_research_rules()}"
        ),
        working_memory_template=load_strategy_working_memory_template(
            AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR
        ),
    )


RESEARCH_PROFILE = build_research_profile()
