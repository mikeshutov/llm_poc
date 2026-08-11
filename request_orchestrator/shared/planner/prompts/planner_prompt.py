import json

from request_orchestrator.agents.models.agent_profile import PROFILE_MANAGEMENT_AGENT_NAME
from request_orchestrator.constants import PLANNER_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, PreviousIteration, PreviousIterationStep
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.planner.models.compiled_planner_context import CompiledPlannerContext
from request_orchestrator.shared.planner.prompts.planner_rules import build_planner_rules
from request_orchestrator.shared.planner.prompts.planner_schema_prompt import PLANNER_SCHEMA


ATTRIBUTE_TYPE_MINIMAL_DESCRIPTION = 'Typed user-attribute key such as `food.likes`, `projects.goals`, or `technology.skills`.'


def _is_profile_management_agent(state: AgentState) -> bool:
    return state.agent_profile.name == PROFILE_MANAGEMENT_AGENT_NAME


def _format_field_line(name: str, field_info, required: bool, minimal: bool = False) -> str:
    description = getattr(field_info, 'description', None) or 'No description provided.'
    if minimal and name == 'attribute_type':
        description = ATTRIBUTE_TYPE_MINIMAL_DESCRIPTION
    requirement = 'required' if required else 'optional'
    return f"- {name} ({requirement}): {description}"


def _format_minimal_tool_schema(tool) -> str:
    args_schema = getattr(tool, 'args_schema', None)
    lines = [f"- {tool.name}"]

    if args_schema is not None:
        required_fields = []
        optional_fields = []
        for field_name, field_info in args_schema.model_fields.items():
            if field_info.is_required():
                required_fields.append(_format_field_line(field_name, field_info, required=True, minimal=True))
            else:
                optional_fields.append(_format_field_line(field_name, field_info, required=False, minimal=True))

        if required_fields:
            lines.append('  Required fields:')
            lines.extend(f"  {line}" for line in required_fields)
        if optional_fields:
            lines.append('  Optional fields:')
            lines.extend(f"  {line}" for line in optional_fields)

    return "\n".join(lines)


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
            if category.rules and not _is_profile_management_agent(state):
                rules[category_name] = category.rules
    else:
        for category in allowed_categories.values():
            tools.extend(category.tools)
            if category.rules and not _is_profile_management_agent(state):
                rules_name = next((name for name, candidate in allowed_categories.items() if candidate is category), None)
                if rules_name is not None:
                    rules[rules_name] = category.rules

    tools.extend(state.agent_profile.extra_tools)

    deduped_tools: dict[str, object] = {}
    for tool in tools:
        deduped_tools[getattr(tool, 'name')] = tool

    if _is_profile_management_agent(state):
        compiled_tools = "\n".join(_format_minimal_tool_schema(tool) for tool in deduped_tools.values())
    else:
        compiled_tools = "\n".join(f"- {t.name}: {t.description}".strip() for t in deduped_tools.values())
    return CompiledPlannerContext(tools=list(deduped_tools.values()), compiled_tools=compiled_tools, rules=rules)


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

    compiled_rules = build_planner_rules(context.rules)
    if state.agent_profile.planner_rules:
        compiled_rules = f"{compiled_rules}\n\nAgent Rules:\n{state.agent_profile.planner_rules}"

    prompt = AgentPrompt(
        prompt_kind=PLANNER_PROMPT_KIND,
        instruction=state.agent_profile.planner_instruction,
        user_profile=state.user_profile,
        task=_build_planner_task(state),
        latest_user_prompt=state.task,
        available_tools=context.compiled_tools,
        rules=compiled_rules,
        previous_iterations=previous_iterations,
        schema=PLANNER_SCHEMA,
    )
    prompt.include_user_profile(include_management_fields=_is_profile_management_agent(state))
    prompt.include_conversation_context()
    prompt.include_available_tools()
    prompt.include_rules_raw()
    prompt.include_previous_iterations()
    prompt.include_latest_user_prompt()
    prompt.include_schema_raw()
    prompt.include_task(heading="Task:" if _is_profile_management_agent(state) else "Goal:")
    return prompt
