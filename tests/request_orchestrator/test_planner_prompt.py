from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules["pycountry"] = pycountry_module

from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.models.agent_prompt import PromptSectionKeys
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt


def test_planner_prompt_exposes_top_level_evidence_views_not_tool_results() -> None:
    state = AgentState.new(task="Find a good answer", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
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
    state.gather_tool_results = lambda: [
        ToolResult(
            tool_call_id=uuid4(),
            plan_step_id=plan.steps[0].db_id,
            tool_name="generic_web_search",
            result={"secret": "raw payload should not be in planner prompt"},
            evidence=[
                EvidenceView(
                    item_id="item-1",
                    tool_name="generic_web_search",
                    title="Example Result",
                    summary="Short evidence summary.",
                    source="generic_web_search",
                    entity_type="web_search_results",
                    llm_metadata={"kind": "web"},
                )
            ],
        )
    ]

    prompt = build_planner_prompt(state)
    evidence_section = prompt.to_log_input_object()["sections_raw"][PromptSectionKeys.EVIDENCE]

    assert evidence_section[0]["type"] == "web_search_results"
    assert evidence_section[0]["evidence"][0]["title"] == "Example Result"
    assert evidence_section[0]["evidence"][0]["summary"] == "Short evidence summary."
    assert evidence_section[0]["evidence"][0]["metadata"] == {"kind": "web"}
    assert "secret" not in str(evidence_section)
