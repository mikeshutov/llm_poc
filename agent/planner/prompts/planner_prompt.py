import json

from agent.agentstate.model import AgentState
from agent.planner.models.compiled_planner_context import CompiledPlannerContext
from agent.planner.prompts.planner_rules import build_planner_rules
from agent.planner.prompts.planner_schema_prompt import PLANNER_SCHEMA
from agent.prompts.agent_prompt import AgentPrompt, PreviousIteration, PreviousIterationStep
from conversation.models.conversation_models import ConversationContext
from tool.tools import TOOL_CATEGORIES, tools as all_tools


# we build the tools and rules dynamically here based on the state before we pass this on to the planner
def _compile_tools_rules_from_state(state: AgentState) -> CompiledPlannerContext:
    applicable = state.request_analysis.applicable_tool_categories
    if applicable:
        tools = []
        rules = {}
        for cat in applicable:
            if cat in TOOL_CATEGORIES:
                tools.extend(TOOL_CATEGORIES[cat].tools)
                if TOOL_CATEGORIES[cat].rules:
                    rules[cat] = TOOL_CATEGORIES[cat].rules
    else:
        tools = all_tools
        rules = {}
    compiled_tools = "\n".join(f"- {t.name}: {t.description}".strip() for t in tools)
    return CompiledPlannerContext(tools=tools, compiled_tools=compiled_tools, rules=rules)


def _build_planner_context(state: AgentState) -> ConversationContext:
    return ConversationContext(
        tool_summary=state.conversation_context.tool_summary,
        recent_roundtrip_tool_summaries=state.conversation_context.recent_roundtrip_tool_summaries,
    )


def _build_planner_task(state: AgentState) -> str:
    goal = state.request_analysis.goal.strip()
    return goal or state.task


def build_planner_prompt(state: AgentState) -> AgentPrompt:
    context = _compile_tools_rules_from_state(state)
    previous_iterations: list[PreviousIteration] = []

    if state.iteration_trace:
        for i, it in enumerate(state.iteration_trace, start=1):
            if it.plan is None:
                previous_iterations.append(
                    PreviousIteration(
                        iteration=i,
                        has_plan=False,
                        steps=[],
                    )
                )
                continue

            steps: list[PreviousIterationStep] = []
            for step in it.plan.steps:
                steps.append(
                    PreviousIterationStep(
                        step_id=step.id,
                        plan=step.plan,
                        tool=step.tool,
                        args=step.args,
                        result=it.results.get(step.id),
                    )
                )

            previous_iterations.append(
                PreviousIteration(
                    iteration=i,
                    has_plan=True,
                    steps=steps,
                )
            )

    return AgentPrompt(
        prompt_kind="planner",
        instruction="You are a planning agent. Utilize data from 'Previous Iterations:' when it is provided.",
        conversation_context=_build_planner_context(state),
        geometadata=state.geometadata,
        task=_build_planner_task(state),
        available_tools=context.compiled_tools,
        rules=build_planner_rules(context.rules),
        previous_iterations=previous_iterations,
        schema=PLANNER_SCHEMA,
    )
