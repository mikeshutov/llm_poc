from __future__ import annotations

import re
from pathlib import PurePosixPath
from urllib.parse import unquote
from uuid import UUID

from agent_runtime.models.agent_profile import AgentName, AgentProfile
from agent_runtime.models.agent_inputs import AgentInputs
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.main_state import MainState
from agent_runtime.models.orchestrator_result import OrchestratorResult
from agent_runtime.models.synthesized_result import SynthesisResult, SynthesisResultBlock
from agent_runtime.working_memory import initialize_working_memory_file
from agent_runtime.context import bind_agent_context
from agents.main.agent import MainReasonerStrategy
from capabilities.adapters.math import calculate
from capabilities.adapters.profile import (
    USER_MANAGEMENT_TOOL,
)
from capabilities.discovery.discover_capabilities_tool import (
    CAPABILITY_DISCOVERY_TOOL,
)
from capabilities.registry.constants import TOOL_CATEGORY_FILE_MANAGEMENT, TOOL_CATEGORY_WRITING
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from retrieval.evidence_store import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
    filter_evidence_steps,
)
from agent_runtime.synthesis.synthesis import _build_result_blocks
from assets.files.urls import STATIC_FILES_URL_PREFIX


_GENERATED_FILE_LINK = re.compile(
    rf"\]\(\s*(?:(?:sandbox:)?{re.escape(STATIC_FILES_URL_PREFIX)}|sandbox:/mnt/data/)"
    r"(?P<path>[^)\s]+)\s*\)",
    re.IGNORECASE,
)


def _file_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _normalize_generated_file_links(
    result_blocks: list[SynthesisResultBlock],
    evidence_steps,
) -> list[SynthesisResultBlock]:
    """Replace model-invented sandbox links with canonical file evidence URLs."""
    file_urls: dict[str, str] = {}
    for step in evidence_steps:
        for evidence in step.evidence:
            urls = evidence.data.get("urls", [])
            evidence_url = next((entry.get("url", "") for entry in urls if isinstance(entry, dict) and entry.get("url")), "")
            if evidence.type != "file" or not evidence_url:
                continue
            file_urls[_file_name_key(str(evidence.data.get("title", "")))] = evidence_url

    fallback_url = next(iter(file_urls.values())) if len(file_urls) == 1 else ""

    def replace_link(match: re.Match[str]) -> str:
        filename = PurePosixPath(unquote(match.group("path"))).name
        canonical_url = file_urls.get(_file_name_key(filename), fallback_url)
        # Never expose model- or sandbox-generated filesystem links. A
        # downloadable link is valid only when it came from file evidence.
        return f"]({canonical_url})" if canonical_url else "]"

    return [
        block.model_copy(update={"content": _GENERATED_FILE_LINK.sub(replace_link, block.content)})
        for block in result_blocks
    ]


TOP_LEVEL_PROFILE = AgentProfile(
    name=AgentName.REQUEST_ORCHESTRATOR,
    scope=MAIN_AGENT_MODEL_SCOPE,
    version=2,
    description="Top-level request reasoner, coordinator, and executor.",
    excluded_categories={TOOL_CATEGORY_FILE_MANAGEMENT, TOOL_CATEGORY_WRITING},
    static_tools=[CAPABILITY_DISCOVERY_TOOL, USER_MANAGEMENT_TOOL, calculate],
    planner_instruction=(
        "For straightforward arithmetic or percentage questions, call math_calculate "
        "directly when a calculation tool is available; do not delegate these requests "
        "to the coding agent or use code execution."
    ),
)
TOP_LEVEL_STRATEGY = MainReasonerStrategy()


def initialize_request_agent_state(main_state: MainState) -> AgentState:
    if main_state.main_agent_state is None:
        main_state.main_agent_state = AgentState.new(
            agent_profile=TOP_LEVEL_PROFILE,
            inputs=AgentInputs.new(task=main_state.task, request_task=main_state.task),
            execution_context=main_state.execution_context,
            llm=main_state.llm,
        )
    return main_state.main_agent_state


def run_main_request_strategy(main_state: MainState) -> MainState:
    execution_state = initialize_request_agent_state(main_state)
    execution_state.inputs.discovered_capabilities = main_state.discovered_capabilities
    execution_state.inputs.discovered_attribute_types = list(main_state.discovered_attribute_types)
    execution_state.available_agent_states = main_state.agent_states
    initialize_working_memory_file(execution_state)
    with bind_agent_context(execution_state):
        execution_state = TOP_LEVEL_STRATEGY.run(
            execution_state, thread_id=execution_state.execution_context.conversation_id or "",
        )
    main_state.discovered_capabilities = execution_state.inputs.discovered_capabilities
    main_state.agent_states = execution_state.available_agent_states or {}
    main_state.main_agent_state = execution_state
    main_state.relevant_evidence_ids = list(execution_state.result.relevant_evidence_ids)
    main_state.plan = execution_state.node_states.planner.plan
    main_state.missing_information = []
    synthesis_result = execution_state.reasoner_result
    if synthesis_result is None:
        synthesis_result = SynthesisResult(
            result=[SynthesisResultBlock(content="I couldn't complete that request.")],
            next_question="Would you like me to try a different search or provide more details?",
            roundtrip_summary="The reasoner did not produce a final answer.",
        )
    evidence_bundle = build_evidence_bundle_from_tool_results(main_state.gather_tool_results())
    evidence_steps = build_evidence_steps_from_tool_results(
        main_state.gather_tool_results(),
        evidence_bundle.evidence_views_by_tool_call_id,
    )
    evidence_steps = filter_evidence_steps(evidence_steps, set(main_state.relevant_evidence_ids)) or evidence_steps
    result_blocks, used_evidence_ids = _build_result_blocks(synthesis_result, evidence_steps)
    result_blocks = _normalize_generated_file_links(result_blocks, evidence_steps)
    main_state.result = OrchestratorResult(
        agent_result=main_state.result.agent_result.copy(
            tool_call_ids=main_state.gather_tool_call_ids(),
            relevant_evidence_ids=list(main_state.relevant_evidence_ids),
        ),
        result_blocks=result_blocks,
        answer=[block.content for block in result_blocks],
        next_question=synthesis_result.next_question,
        roundtrip_summary=synthesis_result.roundtrip_summary,
        roundtrip_latency_ms=main_state.result.roundtrip_latency_ms,
    )
    main_state.relevant_evidence_ids = [UUID(value) for value in used_evidence_ids if _is_uuid(value)]
    return main_state


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False
