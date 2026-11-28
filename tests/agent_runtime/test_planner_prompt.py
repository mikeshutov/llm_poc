from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest


from agents.main.request import TOP_LEVEL_PROFILE
from agents.research.profile import RESEARCH_PROFILE
from agents.delegate.profile import DELEGATE_PROFILE
from capabilities.registry.constants import (
    TOOL_CATEGORY_FILE_MANAGEMENT,
    TOOL_CATEGORY_MATH,
)
from capabilities.registry.tools import WORKING_MEMORY_TOOLS
from agent_runtime.models.agent_prompt import PromptSectionKeys
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from retrieval.evidence import EvidenceView, ToolResult
from agent_runtime.models.plan import Plan
from agent_runtime.planning.prompts.planner_prompt import build_planner_prompt
from capabilities.discovery.models import AvailableTool, CapabilityDiscoveryResult
from personalization.profile.models import UserProfile


def test_planner_prompt_exposes_top_level_evidence_views_not_tool_results(agent_state_factory) -> None:
    state = agent_state_factory(task="Find a good answer", llm=object(), profile=TOP_LEVEL_PROFILE)
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


def test_planner_prompt_explains_an_empty_execution_retry(agent_state_factory) -> None:
    state = agent_state_factory(task="Find a good answer", llm=object(), profile=TOP_LEVEL_PROFILE)
    state.node_states.planner.no_result_attempts = 1

    prompt, _ = build_planner_prompt(state)

    assert "The previous plan execution produced no results." in prompt.build()


def test_top_level_planner_prompt_includes_shared_rules(agent_state_factory) -> None:
    state = agent_state_factory(task="Find a good answer", llm=object(), profile=TOP_LEVEL_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "Each tool call must include an affirmative, factual, testable hypothesis" in prompt.build()


def test_plan_execute_evaluate_planner_batches_independent_tools(agent_state_factory) -> None:
    state = agent_state_factory(task="Find a good answer", llm=object(), profile=TOP_LEVEL_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "include all of them in the same plan" in prompt.build()


def test_planner_prompt_includes_evaluator_missing_information(agent_state_factory) -> None:
    state = agent_state_factory(task="Find a good answer", llm=object(), profile=TOP_LEVEL_PROFILE)
    state.node_states.evaluator.missing_information = ["Current Canadian pricing", "Shipping availability"]

    prompt, _ = build_planner_prompt(state)

    assert prompt.sections_raw[PromptSectionKeys.MISSING_INFORMATION] == [
        "Current Canadian pricing",
        "Shipping availability",
    ]
    assert "The evaluator identified unresolved missing_information below." in prompt.build()


def test_planner_prompt_omits_missing_information_guidance_without_evaluator_feedback(
    agent_state_factory,
) -> None:
    state = agent_state_factory(task="Find a good answer", llm=object(), profile=TOP_LEVEL_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "The evaluator identified unresolved missing_information below." not in prompt.build()


def test_research_coordinator_prompt_explains_subproblem_delegation(agent_state_factory) -> None:
    state = agent_state_factory(task="Research a broad topic", llm=object(), profile=RESEARCH_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert "Delegate independent research questions when useful." in prompt.build()
    assert "Give each delegate a focused task" in prompt.build()


def test_delegate_leaf_prompt_keeps_the_assigned_scope(agent_state_factory) -> None:
    state = agent_state_factory(task="Research one subquestion", llm=object(), profile=DELEGATE_PROFILE)

    prompt, _ = build_planner_prompt(state)

    assert not DELEGATE_PROFILE.can_delegate_agents
    assert "Stay within its scope" in prompt.build()
    assert "Record the handoff for the parent in working memory" in prompt.build()


@pytest.mark.parametrize("profile", [RESEARCH_PROFILE, DELEGATE_PROFILE])
@pytest.mark.parametrize("renamed", [False, True])
def test_planner_uses_profile_instructions_without_agent_name_overrides(
    profile,
    renamed,
    agent_state_factory,
):
    profile = replace(
        profile,
        name=f"custom_{profile.name}" if renamed else profile.name,
        planner_instruction="Follow only the instructions configured on this profile.",
    )
    state = agent_state_factory(profile=profile)

    prompt, _ = build_planner_prompt(state)

    assert prompt.instruction == profile.planner_instruction


def test_delegate_prompt_does_not_include_redundant_decision_contract(agent_state_factory) -> None:
    from agents.delegate.profile import DELEGATE_PROFILE
    from agent_runtime.strategies.react.subagent_prompt import build_subagent_decision_prompt

    state = agent_state_factory(task="Research one subquestion", llm=object(), profile=DELEGATE_PROFILE)
    prompt, _ = build_subagent_decision_prompt(state)

    assert "OUTPUT CONTRACT" not in prompt.build()
    assert "Native tool calls are the authoritative actions." in prompt.build()


def test_react_subagent_prompt_excludes_user_profile(agent_state_factory) -> None:
    from agent_runtime.strategies.react.subagent_prompt import build_subagent_decision_prompt

    state = agent_state_factory(
        task="Research one subquestion",
        llm=object(),
        profile=DELEGATE_PROFILE,
        execution_context=AgentExecutionContext.new(
            user_profile=UserProfile(display_name="Should not be sent")
        ),
    )

    prompt, _ = build_subagent_decision_prompt(state)

    assert PromptSectionKeys.USER_PROFILE not in prompt.sections_raw
    assert "Should not be sent" not in prompt.build()


def test_delegate_prompt_requires_periodic_concise_update_memory_states(agent_state_factory) -> None:
    from agents.delegate.profile import DELEGATE_PROFILE

    state = agent_state_factory(task="Research one subquestion", llm=object(), profile=DELEGATE_PROFILE)

    planner_text = build_planner_prompt(state)[0].build()

    assert "At the start of every reasoning cycle" in planner_text
    assert "current understanding" in planner_text
    assert "update_memory_state" in planner_text
    assert "current roundtrip" in planner_text
    assert "At the start of every reasoning cycle" in planner_text
    assert "final available tool-result batch" in planner_text
    assert "task_completed=true" in planner_text
    assert "must not be passed" not in planner_text
    assert "Pass the evidence IDs summarized" not in planner_text


def test_delegate_prompt_injects_runtime_working_memory_contents(working_memory_state_factory) -> None:
    state = working_memory_state_factory(
        task="Research one subquestion",
        content="# Working Memory\n\n## Current State\nThe first source is validated.",
    )

    prompt, _ = build_planner_prompt(state)
    system_message = prompt.build_messages()[0].content

    assert "WORKING MEMORY" in system_message
    assert "The first source is validated." in system_message


def test_delegate_react_prompt_injects_working_memory_into_system_message(
    working_memory_state_factory,
) -> None:
    from agent_runtime.strategies.react.subagent_prompt import build_subagent_decision_prompt

    state = working_memory_state_factory(
        task="Research one subquestion",
        content="# Working Memory\n\n## Decisions\nUse the primary source.",
    )

    prompt, _ = build_subagent_decision_prompt(state)
    system_message = prompt.build_messages()[0].content

    assert "WORKING MEMORY" in system_message
    assert "Use the primary source." in system_message


def test_delegate_prompt_retains_discovered_tools_for_execution(agent_state_factory) -> None:
    state = agent_state_factory(
        task="Fetch the source page",
        llm=object(),
        profile=DELEGATE_PROFILE,
    )
    state.inputs.discovered_capabilities = CapabilityDiscoveryResult(
        tools=[AvailableTool(name="web_fetch", category="web")]
    )

    prompt, tools = build_planner_prompt(state)

    assert "web_fetch" in {tool.name for tool in tools}
    assert "web_fetch" in prompt.build()


def test_discovery_only_exposes_tools_permitted_by_the_profile(agent_profile_factory, agent_state_factory):
    state = agent_state_factory(profile=agent_profile_factory(
        name="restricted",
        scope="main_agent",
        allowed_categories={TOOL_CATEGORY_MATH, TOOL_CATEGORY_FILE_MANAGEMENT},
        excluded_categories={TOOL_CATEGORY_FILE_MANAGEMENT},
        static_tools=list(WORKING_MEMORY_TOOLS),
    ))
    state.inputs.discovered_capabilities = CapabilityDiscoveryResult(tools=[
        AvailableTool(name="math_calculate", category="math"),
        AvailableTool(name="file_delete", category="math"),
        AvailableTool(name="unknown_tool", category="math"),
    ])

    _, tools = build_planner_prompt(state)

    assert {tool.name for tool in tools} == {"math_calculate", "update_memory_state"}


