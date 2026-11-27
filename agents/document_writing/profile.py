from __future__ import annotations

from agent_runtime.models.agent_profile import AgentDelegationArgs
from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentProfile,
    AgentName,
)


class DocumentWritingDelegationArgs(AgentDelegationArgs):
    """LLM-facing parameters for delegating work to the document writer."""
from agents.document_writing.prompts import build_document_writing_rules
from capabilities.registry.constants import (
    TOOL_NAME_FILE_LIST,
    TOOL_NAME_FILE_READ,
    TOOL_NAME_LOOKUP_EVIDENCE,
    TOOL_NAME_WEB_FETCH,
    TOOL_NAME_WRITING_TEMPLATES,
    TOOL_NAME_WORKSPACE_FILE_EDIT,
    TOOL_NAME_WORKSPACE_FILE_LIST,
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    TOOL_NAME_WORKSPACE_FILE_READ,
    TOOL_NAME_WORKSPACE_FILE_WRITE,
)
from capabilities.registry.tools import (
    FILE_MANAGEMENT_TOOLS,
    MEMORY_TOOLS,
    TOOL_CATEGORIES,
    WEB_SEARCH_TOOLS,
    WRITING_TOOLS,
    WORKSPACE_TOOLS,
    WORKING_MEMORY_TOOLS,
)
from agent_runtime.strategies.working_memory import load_strategy_working_memory_template
from agent_runtime.planning.prompts.planner_rules import WORKING_MEMORY_LIFECYCLE_RULE
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE


_DOCUMENT_WRITING_TOOL_NAMES = {
    TOOL_NAME_FILE_READ,
    TOOL_NAME_FILE_LIST,
    TOOL_NAME_LOOKUP_EVIDENCE,
    TOOL_NAME_WEB_FETCH,
    TOOL_NAME_WRITING_TEMPLATES,
    TOOL_NAME_WORKSPACE_FILE_EDIT,
    TOOL_NAME_WORKSPACE_FILE_LIST,
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    TOOL_NAME_WORKSPACE_FILE_READ,
    TOOL_NAME_WORKSPACE_FILE_WRITE,
}


def build_document_writing_profile() -> AgentProfile:
    static_tools = [
        tool
        for tool in [*FILE_MANAGEMENT_TOOLS, *WORKSPACE_TOOLS, *WORKING_MEMORY_TOOLS, *MEMORY_TOOLS, *WEB_SEARCH_TOOLS, *WRITING_TOOLS]
        if tool.name in _DOCUMENT_WRITING_TOOL_NAMES
    ]
    return AgentProfile(
        name=AgentName.DOCUMENT_WRITING,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=8,
        description=(
            "Create and update documents in the conversation workspace, then promote completed "
            "documents into managed artifacts. This agent supports document formats only: "
            "TXT, Markdown (.md), DOCX, and PDF. Use it for explicit requests to write, "
            "create, edit, rename, or otherwise maintain those document files. Do not use it "
            "for HTML pages, CSS, JavaScript, or other source-code artifacts; route those to "
            "the coding agent. Make sure to provide this agent with an array of Evidence ID's "
            "whenever those are available as well as the goal. The delegation parameters "
            "are task and evidence_ids."
        ),
        allowed_categories=set(),
        excluded_categories=set(TOOL_CATEGORIES),
        static_tools=[*static_tools, *WORKING_MEMORY_TOOLS],
        execution_strategy=AgentExecutionStrategy.REACT,
        delegatable=True,
        max_turns=10,
        planner_instruction=(
            f"{WORKING_MEMORY_LIFECYCLE_RULE}\n\n"
            "Manage the user's explicit document-writing request using the available "
            "workspace and managed-file tools. Do not answer the user directly.\n\n"
            f"{build_document_writing_rules()}"
        ),
        working_memory_template=load_strategy_working_memory_template(AgentExecutionStrategy.REACT),
        delegation_args_schema=DocumentWritingDelegationArgs,
    )


DOCUMENT_WRITING_PROFILE = build_document_writing_profile()
