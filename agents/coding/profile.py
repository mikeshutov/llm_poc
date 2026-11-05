from __future__ import annotations

from agent_runtime.models.agent_profile import (
    AgentDelegationArgs,
    AgentExecutionStrategy,
    AgentProfile,
    AgentName,
)
from capabilities.adapters.math import calculate
from capabilities.adapters.finance import get_crypto_historical_prices
from capabilities.registry.constants import (
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
)
from capabilities.registry.tools import MEMORY_TOOLS
from capabilities.registry.tools import WORKSPACE_TOOLS
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
}


def build_coding_profile() -> AgentProfile:
    static_tools = [
        tool
        for tool in (
            calculate,
            get_crypto_historical_prices,
            *MEMORY_TOOLS,
            *WORKSPACE_TOOLS,
        )
        if tool.name in _CODING_TOOL_NAMES
    ]
    return AgentProfile(
        name=AgentName.CODING,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=5,
        description=(
            "An agent capable of generating, executing, and validating code to solve "
            "computational tasks."
        ),
        allowed_categories={"math", "finance", "finance_crypto", "code_execution"},
        static_tools=static_tools,
        execution_strategy=AgentExecutionStrategy.REACT,
        delegatable=True,
        max_turns=10,
        planner_instruction=(
            "Solve the coding or computational task using iterative reasoning and the "
            "available capabilities. Generate clear code when code is requested, execute it "
            "only through the sandboxed code-execution capability, and validate outputs before "
            "using memory_lookup_evidence when prior evidence IDs are supplied or relevant. "
            "reporting completion. The execution workspace is mounted at /workspace; do not use or assume "
            "paths such as /mnt/data. Persist durable text, Markdown, DOCX, or PDF artifacts "
            "through workspace files, then promote completed artifacts into the user's managed files. Supported durable "
            "formats include text, Markdown, CSV, SVG, DOCX, and PDF.\n\n"
            "Coding rules:\n"
            "- Treat the task as a general computational or coding problem, not only a software engineering task.\n"
            "- When a request is underspecified, use a safe, conventional, reversible default and state the assumption. Ask only when the missing detail materially changes the outcome, requires user-provided data or authorization, creates a safety or privacy concern, or has no reasonable default. For example, use 1 through 10 for an unspecified multiplication table and generate a clearly labeled dataset for an explicit sample chart. For relative time such as 'past 12 months', use the trailing period ending today; do not substitute arbitrary calendar years.\n"
            "- Prefer small, testable steps and validate assumptions before finalizing code.\n"
            "- Use workspace tools for source code and intermediate artifacts; use workspace_file_promote only for the final artifact.\n"
            "- Use calculation for supported arithmetic and code_execute for Python execution.\n"
            "- Never claim code was executed or validated by a runtime without a code_execute result.\n"
            "- Use memory_lookup_evidence with evidence IDs when prior conversation evidence is needed; "
            "do not invent evidence details from an ID alone.\n"
            "- When evidence_ids are supplied for a data task, retrieve them once and use the returned "
            "metadata/observations directly; do not repeat the lookup or replace the data with a new "
            "network request.\n"
            "- The code execution workspace is conversation-scoped at /workspace; use workspace tools for source and intermediate files.\n"
            "- Assume only the Python standard library is available unless execution output proves a dependency is installed.\n"
            "- Before generating code that imports a non-standard plotting or data library, verify the dependency in the execution environment; if it is unavailable, use a supported standard-library/vector fallback or report the limitation instead of letting the first full run fail.\n"
            "- code_execute returns execution output but does not persist generated files or return binary artifacts.\n"
            "- Promote supported durable artifacts, including CSV and SVG for simple vector images, only after validation. If the task requires an unsupported binary format or unavailable dependency, report that limitation clearly instead of claiming success."
            "\n- Use workspace file tools for temporary source files and intermediate outputs; use relative paths only."
        ),
        delegation_args_schema=AgentDelegationArgs,
    )


CODING_PROFILE = build_coding_profile()
