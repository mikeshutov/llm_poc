import json

from request_orchestrator.constants import PLANNER_PROMPT_KIND
from request_orchestrator.agents.models.agent_profile import DEFAULT_PLANNER_PROMPT_INSTRUCTION
from request_orchestrator.models.agent_prompt import AgentPrompt, PreviousIteration, PreviousIterationStep
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.planner.models.compiled_planner_context import CompiledPlannerContext
from request_orchestrator.shared.planner.prompts.planner_rules import build_planner_rules
from request_orchestrator.shared.planner.prompts.planner_schema_prompt import PLANNER_SCHEMA


# we build the tools and rules dynamically here based on the state before we pass this on to the planner
def _compile_tools_rules_from_state(state: AgentState) -> CompiledPlannerContext:
    allowed_categories = state.agent_profile.allowed_tool_categories()
    tools = []
    rules = {}

    if state.request_analysis.applicable_tool_categories:
        for category_name in state.request_analysis.applicable_tool_categories:
            category = allowed_categories.get(category_name)
            if category is None:
                continue
            tools.extend(category.tools)
            if category.rules:
                rules[category_name] = category.rules
    else:
        for category in allowed_categories.values():
            tools.extend(category.tools)

    tools.extend(state.agent_profile.extra_tools)

    deduped_tools: dict[str, object] = {}
    for tool in tools:
        deduped_tools[getattr(tool, 'name')] = tool

    compiled_tools = "\n".join(f"- {t.name}: {t.description}".strip() for t in deduped_tools.values())
    return CompiledPlannerContext(tools=list(deduped_tools.values()), compiled_tools=compiled_tools, rules=rules)


def _build_planner_task(state: AgentState) -> str:
    goal = state.request_analysis.goal.strip()
    return goal or state.task


def build_planner_prompt(state: AgentState) -> AgentPrompt:
    context = _compile_tools_rules_from_state(state)
    previous_iterations: list[PreviousIteration] = []
    has_prior_tool_results = any(bool(iteration.results) for iteration in state.iteration_trace)

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

    compiled_rules = build_planner_rules(
        context.rules,
        requires_tools=state.request_analysis.requires_tools,
        has_prior_tool_results=has_prior_tool_results,
    )
    if state.agent_profile.planner_rules:
        compiled_rules = f"{compiled_rules}\n\nAgent Rules:\n{state.agent_profile.planner_rules}"

    return AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction=DEFAULT_PLANNER_PROMPT_INSTRUCTION,
        user_profile=state.user_profile,
        task=_build_planner_task(state),
        latest_user_prompt=state.task,
        available_tools=context.compiled_tools,
        rules=compiled_rules,
        previous_iterations=previous_iterations,
        schema=PLANNER_SCHEMA,
        include_user_attribute_management_fields=state.agent_profile.name == "profile_management",
    )
