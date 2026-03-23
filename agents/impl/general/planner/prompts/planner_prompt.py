from datetime import datetime
from typing import Sequence
from zoneinfo import ZoneInfo

import json

from agents.agentstate.model import AgentState, IterationState
from agents.impl.general.planner.models.compiled_planner_context import CompiledPlannerContext
from agents.impl.general.planner.prompts.planner_rules import build_planner_rules
from agents.impl.general.planner.prompts.planner_schema_prompt import PLANNER_SCHEMA
from tool.tools import TOOL_CATEGORIES, tools as all_tools


def current_time_block(*,timezone = "America/Toronto") -> str:
    now = datetime.now(ZoneInfo(timezone))
    return (
        f"Current datetime: {now.isoformat()}\n\n"
        f"Current date: {now.date()}\n"
        f"Current weekday: {now.strftime('%A')}\n"
        f"Timezone: {timezone}"
    )

# we build the tools and rules dynamically here based on the state before we pass this on to the planner
def _compile_tools_rules_from_state(state: AgentState) -> CompiledPlannerContext:
    applicable = state.classification_results.applicable_tool_categories
    if applicable:
        tools, rules = [], {}
        for cat in applicable:
            if cat in TOOL_CATEGORIES:
                tools.extend(TOOL_CATEGORIES[cat].tools)
                if TOOL_CATEGORIES[cat].rules:
                    rules[cat] = TOOL_CATEGORIES[cat].rules
    else:
        # for now just fallback to all tools and no rules
        tools = all_tools
        rules = {}
    compiled_tools = "\n".join(f"- {t.name}: {t.description}".strip() for t in tools)
    return CompiledPlannerContext(tools=tools, compiled_tools=compiled_tools, rules=rules)


def build_planner_prompt(state: AgentState) -> str:
    context = _compile_tools_rules_from_state(state)
    planner_prompt = (
        "You are a planning agent. Utilize data from 'Previous Iterations:' when it is provided.\n\n"
        # Personalization section. For now just a time prompt.
        f"{current_time_block()}\n\n"
        "TASK:\n"
        f"{state.task}\n"
        f"Allowed Tools:\n{context.compiled_tools}\n\n"
        f"{build_planner_rules(context.rules)}\n\n"
    )
    
    # Build append previous call plan and results.
    if state.iteration_trace:
            blocks: list[str] = []
            for i, it in enumerate(state.iteration_trace, start=1):
                if it.plan is None:
                    blocks.append(f"Iteration {i}:\n  (no plan)\n")
                    continue

                step_lines: list[str] = []
                for step in it.plan.steps:
                    evidence = it.results.get(step.id, None)
                    step_lines.append(
                        "\n".join([
                            f"  - step_id: {step.id}",
                            f"    plan: {step.plan}",
                            f"    tool: {step.tool}",
                            f"    args: {json.dumps(step.args)}",
                            f"    result[{step.id}]: {evidence}",
                        ])
                    )

                blocks.append(f"Iteration {i}:\n" + "\n".join(step_lines))

            planner_prompt += "\n\nPrevious Iterations:\n" + "\n\n".join(blocks) + "\n"

    planner_prompt+= f"{PLANNER_SCHEMA}\n\n"

    return planner_prompt
