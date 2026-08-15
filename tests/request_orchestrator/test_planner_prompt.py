from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules["pycountry"] = pycountry_module

from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.models.agent_prompt import PromptSectionKeys
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.models.plan import Plan
from request_orchestrator.models.plan_step_ids import format_plan_step_id
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt


def test_planner_prompt_exposes_top_level_evidence_views_not_tool_results() -> None:
    state = AgentState.new(task="Find a good answer", max_turns=3, llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    plan = Plan.model_validate(
        {
            "steps": [
                {
                    "id": "E1",
                    "plan": "Search for results",
                    "tool": "generic_web_search",
                    "args": {"query_text": "example query"},
                }
            ]
        }
    )
    state.node_states.planner.plan = plan
    state.node_states.planner.plan_count = 1
    state.result = state.result.copy(tool_results=[
        ToolResult(
            step_id=format_plan_step_id(1, "E1"),
            tool_name="generic_web_search",
            iteration=1,
            result={"secret": "raw payload should not be in planner prompt"},
            evidence_views=[
                EvidenceView(
                    item_id="item-1",
                    title="Example Result",
                    summary="Short evidence summary.",
                    metadata={"kind": "web"},
                    evidence_object={"detail": "should stay out of planner"},
                )
            ],
            hydrated_evidence=[
                HydratedEvidence(
                    item_id="item-1",
                    tool_name="generic_web_search",
                    title="Example Result",
                    summary="Short evidence summary.",
                    source="generic_web_search",
                    entity_type="web_search_results",
                    metadata={"kind": "web"},
                )
            ],
        )
    ])

    prompt = build_planner_prompt(state)
    evidence_section = prompt.get_section_content(PromptSectionKeys.EVIDENCE)

    assert '"type": "web_search_results"' in evidence_section
    assert '"metadata": {' in evidence_section
    assert '"title": "Example Result"' in evidence_section
    assert '"summary": "Short evidence summary."' in evidence_section
    assert '"evidence_object"' in evidence_section
    assert "should stay out of planner" in evidence_section
    assert "raw payload should not be in planner prompt" not in evidence_section
