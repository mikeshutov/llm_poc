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

from agents.main.request import TOP_LEVEL_PROFILE
from agents.research.profile import RESEARCH_PROFILE
from agent_runtime.models.agent_prompt import PromptSectionKeys
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from retrieval.evidence import EvidenceView, ToolResult
from agent_runtime.models.plan import Plan
from agent_runtime.planning.prompts.planner_prompt import build_planner_prompt


def test_planner_prompt_exposes_top_level_evidence_views_not_tool_results() -> None:
    state = AgentState.new(task="Find a good answer", llm=object(), agent_profile=TOP_LEVEL_PROFILE)
    plan = Plan.model_validate(
        {
            "steps": [
                {
                    "id": "E1",
                    "plan": "Search for results",
                    "tool": "web_search",
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
                tool_name="web_search",
            result={"secret": "raw payload should not be in planner prompt"},
            evidence=[
                EvidenceView(
                    summary="Short evidence summary.",
                        source="web_search",
        type="web_search_results",
        metadata={"kind": "web"},
        data={
            "item_id": "item-1",
            "title": "Example Result",
        })
            ],
        )
    ]

    prompt, _ = build_planner_prompt(state)
    evidence_section = prompt.sections_raw[PromptSectionKeys.EVIDENCE]

    assert evidence_section[0]["type"] == "web_search_results"
    assert evidence_section[0]["evidence"][0]["data"]["title"] == "Example Result"
    assert evidence_section[0]["evidence"][0]["summary"] == "Short evidence summary."
    assert evidence_section[0]["evidence"][0]["metadata"] == {"kind": "web"}
    assert "secret" not in str(evidence_section)

    messages = prompt.build_messages()
    assert all("Example Result" not in message.content for message in messages)


def test_planner_prompt_explains_an_empty_execution_retry() -> None:
    state = AgentState.new(task="Find a good answer", llm=object(), agent_profile=TOP_LEVEL_PROFILE)
    state.node_states.planner.no_result_attempts = 1

    prompt, _ = build_planner_prompt(state)

    assert "The previous plan execution produced no results." in prompt.build()


def test_top_level_planner_prompt_includes_shared_rules() -> None:
    state = AgentState.new(task="Find a good answer", llm=object(), agent_profile=TOP_LEVEL_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "Each tool call must include an affirmative, factual, testable hypothesis" in prompt.build()


def test_plan_execute_evaluate_planner_batches_independent_tools() -> None:
    state = AgentState.new(task="Find a good answer", llm=object(), agent_profile=TOP_LEVEL_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "include all of them in the same plan" in prompt.build()


def test_planner_prompt_includes_evaluator_missing_information() -> None:
    state = AgentState.new(task="Find a good answer", llm=object(), agent_profile=TOP_LEVEL_PROFILE)
    state.node_states.evaluator.missing_information = ["Current Canadian pricing", "Shipping availability"]

    prompt, _ = build_planner_prompt(state)

    assert prompt.sections_raw[PromptSectionKeys.MISSING_INFORMATION] == [
        "Current Canadian pricing",
        "Shipping availability",
    ]
    assert "The evaluator identified unresolved missing_information below." in prompt.build()


def test_planner_prompt_omits_missing_information_guidance_without_evaluator_feedback() -> None:
    state = AgentState.new(task="Find a good answer", llm=object(), agent_profile=TOP_LEVEL_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "The evaluator identified unresolved missing_information below." not in prompt.build()


def test_research_coordinator_prompt_explains_subproblem_delegation() -> None:
    state = AgentState.new(task="Research a broad topic", llm=object(), agent_profile=RESEARCH_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "decompose the request into named, substantial subproblems" in prompt.build()
    assert "map which ones are independent versus dependent" in prompt.build()


def test_delegate_leaf_prompt_explains_that_delegation_is_forbidden() -> None:
    from agents.delegate.profile import DELEGATE_PROFILE

    leaf_profile = DELEGATE_PROFILE
    state = AgentState.new(task="Research one subquestion", llm=object(), agent_profile=leaf_profile)

    prompt, _ = build_planner_prompt(state)

    assert "cannot delegate further" in prompt.build()
    assert "Do not answer the user directly" in prompt.build()


def test_delegate_prompt_does_not_include_redundant_decision_contract() -> None:
    from agents.delegate.profile import DELEGATE_PROFILE
    from agent_runtime.strategies.react.subagent_prompt import build_subagent_decision_prompt

    state = AgentState.new(task="Research one subquestion", llm=object(), agent_profile=DELEGATE_PROFILE)
    prompt, _ = build_subagent_decision_prompt(state)

    assert "OUTPUT CONTRACT" not in prompt.build()
    assert "Native tool calls are the authoritative actions." in prompt.build()


def test_delegate_prompt_requires_periodic_concise_working_memory_updates() -> None:
    from agents.delegate.profile import DELEGATE_PROFILE

    state = AgentState.new(task="Research one subquestion", llm=object(), agent_profile=DELEGATE_PROFILE)

    planner_text = build_planner_prompt(state)[0].build()

    assert "After receiving substantial new information from tools" in planner_text
    assert "Consolidate existing findings rather than duplicating them" in planner_text
    assert "Before performing additional research" in planner_text
    assert "not the chronological history of your tool calls" in planner_text


def test_delegate_prompt_injects_runtime_working_memory_contents(tmp_path, monkeypatch) -> None:
    from agents.delegate.memory import initialize_working_memory_file
    from agents.delegate.profile import DELEGATE_PROFILE

    state = AgentState.new(
        task="Research one subquestion",
        llm=object(),
        agent_profile=DELEGATE_PROFILE,
        execution_context=AgentExecutionContext.new(conversation_id="conversation-1"),
    )
    monkeypatch.setattr(
        "agents.delegate.memory.conversation_workspace_path",
        lambda conversation_id, user_id=None: tmp_path,
    )
    monkeypatch.setattr(
        "assets.files.storage.conversation_workspace_path",
        lambda conversation_id, user_id=None: tmp_path,
    )
    initialize_working_memory_file(state)
    (tmp_path / state.working_memory_file_name).write_text(
        "# Working Memory\n\n## Current State\nThe first source is validated.",
        encoding="utf-8",
    )

    prompt, _ = build_planner_prompt(state)
    system_message = prompt.build_messages()[0].content

    assert "The first source is validated." in system_message




