from datetime import datetime
from typing import Sequence
from zoneinfo import ZoneInfo

import json

from agent.agentstate.model import IterationState
from agent.planner.prompts.planner_rules import build_planner_rules
from agent.planner.prompts.planner_schema_prompt import PLANNER_SCHEMA


def current_time_block(*,timezone = "America/Toronto") -> str:
    now = datetime.now(ZoneInfo(timezone))
    return (
        f"Current datetime: {now.isoformat()}\n\n"
        f"Current date: {now.date()}\n"
        f"Current weekday: {now.strftime('%A')}\n"
        f"Timezone: {timezone}"
    )


def build_planner_prompt(*, tool_list: str, task: str, previous_calls: Sequence[IterationState]) -> str:
    planner_prompt =  (
        f"{current_time_block()}\n\n"
        "You are a planning agent. Utilize data from 'Previous Iterations:' when it is provided.\n\n"
        f"Allowed TOOLs: {tool_list}\n\n"
        f"{PLANNER_SCHEMA}\n\n"
        f"{build_planner_rules()}\n\n"
        "TASK:\n"
        f"{task}\n"
    )
    # Build append previous call plan and results.
    if previous_calls:
            blocks: list[str] = []
            for i, it in enumerate(previous_calls, start=1):
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

    return planner_prompt
