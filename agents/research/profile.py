from __future__ import annotations

from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentProfile,
    AgentName,
)
from agents.research.prompts import build_research_rules
from agents.research.prompts import load_research_template
from capabilities.adapters.workspace import (
    workspace_file_edit,
    workspace_file_complete,
    workspace_file_list,
    workspace_file_read,
    workspace_file_write,
)
from capabilities.registry.constants import (
    TOOL_NAME_WORKSPACE_FILE_EDIT,
    TOOL_NAME_WORKSPACE_FILE_COMPLETE,
    TOOL_NAME_WORKSPACE_FILE_LIST,
    TOOL_NAME_WORKSPACE_FILE_READ,
    TOOL_NAME_WORKSPACE_FILE_WRITE,
)
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE


_RESEARCH_WORKSPACE_TOOLS = {
    TOOL_NAME_WORKSPACE_FILE_WRITE: workspace_file_write,
    TOOL_NAME_WORKSPACE_FILE_READ: workspace_file_read,
    TOOL_NAME_WORKSPACE_FILE_EDIT: workspace_file_edit,
    TOOL_NAME_WORKSPACE_FILE_COMPLETE: workspace_file_complete,
    TOOL_NAME_WORKSPACE_FILE_LIST: workspace_file_list,
}


def build_research_profile() -> AgentProfile:
    return AgentProfile(
        name=AgentName.RESEARCH,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=3,
        description=(
            "Research a topic by decomposing it into questions, selecting appropriate "
            "sources, gathering and checking evidence, and writing the final findings to a "
            "research-{topic}.md file in the conversation workspace. The completed report is "
            "the definitive research output for downstream agents."
        ),
        allowed_categories={"web_search", "knowledge", "books", "language", "workspace"},
        static_tools=list(_RESEARCH_WORKSPACE_TOOLS.values()),
        required_completion_tools={TOOL_NAME_WORKSPACE_FILE_COMPLETE},
        execution_strategy=AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR,
        delegatable=True,
        max_turns=8,
        planner_instruction=(
            "Plan and execute a rigorous research workflow for the provided topic. "
            "Decompose the request, gather diverse relevant evidence, and replan when "
            "the evidence leaves an important gap. Write the final report using the supplied "
            "template and make that workspace file the final evidence for downstream agents. "
            "Do not answer the user directly.\n\n"
            "Research template:\n"
            f"{load_research_template()}"
            "\n\n"
            f"{build_research_rules()}"
        ),
    )


RESEARCH_PROFILE = build_research_profile()
