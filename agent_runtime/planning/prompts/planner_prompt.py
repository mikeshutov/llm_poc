from agent_runtime.models.agent_profile import PROFILE_MANAGEMENT_AGENT_NAME
import json

from langchain_core.tools import BaseTool

from agent_runtime.models.agent_prompt import AgentPrompt, PromptSectionKeys
from agent_runtime.models.agent_state import AgentState
from retrieval.evidence_store import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from agent_runtime.planning.models.compiled_planner_context import CompiledPlannerContext
from agent_runtime.planning.prompts.planner_rules import build_planner_rules


ATTRIBUTE_TYPE_MINIMAL_DESCRIPTION = 'Typed user-attribute key such as `food.preference`, `projects.goal`, or `technology.skill`.'
EMPTY_EXECUTION_REPLAN_RULE = (
    "The previous plan execution produced no results. Retry with materially different "
    "tool calls or arguments; do not repeat the same request."
)
MISSING_INFORMATION_REPLAN_RULE = (
    "The evaluator identified unresolved missing_information below. Keep the original task "
    "as the goal, use existing evidence as context, and plan only tool calls that directly "
    "resolve those gaps."
)
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

    result_fields = tuple(getattr(tool, "available_result_fields", ()) or ())
    if result_fields:
        lines.append(f"  Result fields: {', '.join(result_fields)}")

    return "\n".join(lines)


def _group_tools_by_prefix(tools: list[object]) -> list[list[object]]:
    groups: dict[str, list[object]] = {}
    for tool in tools:
        name = str(getattr(tool, "name", ""))
        prefix = name.split(".", 1)[0] if "." in name else name
        groups.setdefault(prefix, []).append(tool)
    return list(groups.values())


def _compile_tools_rules_from_state(state: AgentState) -> CompiledPlannerContext:
    allowed_categories = state.agent_profile.tool_categories
    tools = []
    rules = {}

    agent_tool_categories = list(state.inputs.tool_category_names)
    requested_tool_names = set(state.inputs.tool_names)

    if agent_tool_categories:
        for category_name in agent_tool_categories:
            category = allowed_categories.get(category_name)
            if category is None:
                continue
            tools.extend(category.tools)
            if category.rules:
                rules[category_name] = category.rules
    else:
        for category in allowed_categories.values():
            tools.extend(category.tools)
            if category.rules:
                rules_name = next((name for name, candidate in allowed_categories.items() if candidate is category), None)
                if rules_name is not None:
                    rules[rules_name] = category.rules

    tools.extend(state.agent_profile.static_tools)

    deduped_tools: dict[str, object] = {}
    for tool in tools:
        deduped_tools[getattr(tool, 'name')] = tool

    if requested_tool_names:
        deduped_tools = {
            name: tool
            for name, tool in deduped_tools.items()
            if name in requested_tool_names
        }

    if state.inputs.discovered_capabilities is not None:
        discovered_tool_names = {
            tool.name for tool in state.inputs.discovered_capabilities.tools
        }
        discovered_tool_names.update(tool.name for tool in state.agent_profile.static_tools)
        deduped_tools = {
            name: tool
            for name, tool in deduped_tools.items()
            if name in discovered_tool_names
        }

    active_tool_names = set(deduped_tools)
    for category_name in tuple(rules):
        category = allowed_categories.get(category_name)
        if category is None or not any(tool.name in active_tool_names for tool in category.tools):
            rules.pop(category_name, None)

    from capabilities.registry.tools import TOOLS_BY_NAME

    for tool in deduped_tools.values():
        registered_tool = TOOLS_BY_NAME.get(getattr(tool, "name", ""))
        tool_rules = tuple(
            getattr(tool, "rules", ()) or getattr(registered_tool, "rules", ()) or ()
        )
        if tool_rules:
            rules[getattr(tool, "name")] = list(tool_rules)

    tool_groups = _group_tools_by_prefix(list(deduped_tools.values()))
    if _is_profile_management_agent(state):
        compiled_tools = "\n\n".join(
            "\n".join(_format_minimal_tool_schema(tool) for tool in group)
            for group in tool_groups
        )
    else:
        compiled_tools = "\n\n".join(
            "\n".join(f"- {tool.name}: {tool.description}".strip() for tool in group)
            for group in tool_groups
        )
    return CompiledPlannerContext(tools=list(deduped_tools.values()), compiled_tools=compiled_tools, rules=rules)
def build_planner_prompt(state: AgentState) -> tuple[AgentPrompt, list[BaseTool]]:
    context = _compile_tools_rules_from_state(state)
    if state.available_agent_states is not None:
        capabilities = state.inputs.discovered_capabilities
        prompt = AgentPrompt(
            instruction=(
                "You are the top-level planner. Use the available tools to complete the task. "
                "Do not invent capabilities."
            ),
            user_profile=state.execution_context.user_profile,
            conversation_context=state.execution_context.conversation_context,
            task=state.inputs.task,
            available_tools=(
                [] if capabilities is None else capabilities.tools
            ) + list(state.agent_profile.static_tools),
            rules=build_planner_rules(context.rules),
        )
        for section in (
            PromptSectionKeys.USER_PROFILE,
            PromptSectionKeys.CONVERSATION_CONTEXT,
            PromptSectionKeys.AVAILABLE_TOOLS,
            PromptSectionKeys.RULES,
            PromptSectionKeys.TASK,
        ):
            prompt.include_section(section)
        return prompt, context.tools

    tool_results = state.gather_tool_results()
    evidence_bundle = build_evidence_bundle_from_tool_results(tool_results)
    evidence_steps = build_evidence_steps_from_tool_results(
        tool_results,
        evidence_bundle.evidence_views_by_tool_call_id,
    )

    compiled_rules = build_planner_rules(context.rules)
    if state.agent_profile.planner_rules:
        compiled_rules = f"{compiled_rules}\n\nAgent Rules:\n{state.agent_profile.planner_rules}"
    if state.node_states.planner.no_result_attempts:
        compiled_rules = f"{compiled_rules}\n\nExecution Feedback:\n- {EMPTY_EXECUTION_REPLAN_RULE}"
    if state.node_states.evaluator.missing_information:
        compiled_rules = f"{compiled_rules}\n\nReplanning Guidance:\n- {MISSING_INFORMATION_REPLAN_RULE}"
    discovered_attribute_types = sorted({
        attribute.attribute_type
        for attribute in state.execution_context.user_profile.user_attributes.attributes
    })
    if discovered_attribute_types:
        compiled_rules = (
            f"{compiled_rules}\n\nRelevant user attribute types for this iteration:\n"
            f"- {', '.join(discovered_attribute_types)}"
        )

    prompt = AgentPrompt(
        instruction=state.agent_profile.planner_instruction,
        user_profile=state.execution_context.user_profile,
        conversation_context=state.execution_context.conversation_context,
        task=state.inputs.task,
        available_tools=context.compiled_tools,
        rules=compiled_rules,
        evidence=evidence_steps,
        missing_information=state.node_states.evaluator.missing_information,
    )
    prompt.include_section(
        PromptSectionKeys.USER_PROFILE,
        metadata={
            "include_management_fields": _is_profile_management_agent(state),
            "include_tone": True,
        },
    )
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)
    prompt.include_section(PromptSectionKeys.AVAILABLE_TOOLS)
    prompt.include_section(PromptSectionKeys.RULES)
    prompt.include_section(PromptSectionKeys.EVIDENCE)
    prompt.include_section(PromptSectionKeys.MISSING_INFORMATION)
    prompt.include_section(PromptSectionKeys.TASK)
    return prompt, context.tools
