from __future__ import annotations

import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from common.config import UTF8_ENCODING
from agent_runtime.context import current_agent_state
from agent_runtime.working_memory import MAX_WORKING_MEMORY_BYTES, validate_working_memory, working_memory_path
from capabilities.adapters.workspace.files import _result
from capabilities.registry.constants import TOOL_NAME_UPDATE_MEMORY_STATE
from retrieval.evidence import ToolResult


class UpdateMemoryStateArgs(BaseModel):
    content: str | None = Field(
        default=None,
        description="Complete replacement of the current task state and working memory.",
    )
    find: str | None = Field(
        default=None,
        description="Text or regular expression to replace in the current working memory.",
    )
    replace: str = Field(
        default="",
        description="Replacement text used with find. An empty value removes the match.",
    )
    regex: bool = Field(
        default=False,
        description="Interpret find as a regular expression instead of exact text.",
    )
    append: str | None = Field(
        default=None,
        description="Progress, findings, decisions, or remaining work to append to the current task state.",
    )
    task_completed: bool = Field(
        default=False,
        description=(
            "Set true only on the final successful memory update when the assigned task is complete. "
            "This signals the orchestrator to stop the agent after saving memory."
        ),
    )

    @model_validator(mode="after")
    def validate_update_mode(self) -> "UpdateMemoryStateArgs":
        modes = sum(value is not None for value in (self.content, self.find, self.append))
        if modes != 1:
            raise ValueError("Provide exactly one update mode: content, find, or append.")
        if self.regex and self.find is None:
            raise ValueError("regex=true requires find.")
        return self


@tool(
    TOOL_NAME_UPDATE_MEMORY_STATE,
    args_schema=UpdateMemoryStateArgs,
    description=(
        "Maintain the current agent's task state and working memory: objective, progress, findings, "
        "decisions, supporting evidence IDs, and remaining work. This is how you keep track of the "
        "current task across reasoning cycles, not just a store of remembered facts. "
        f"Keep the saved memory within {MAX_WORKING_MEMORY_BYTES} UTF-8 bytes; consolidate it when needed. "
        "Use content for a full replacement, "
        "find/replace for exact or regex text replacement, or append to add text. "
        "A successful update automatically condenses previously viewed compactable tool exchanges in the "
        "current roundtrip. New results from the current batch remain for the next reasoning cycle. "
        "No source IDs are needed. Non-compactable results cannot be removed. "
        "Use this tool for both durable state updates and completion signaling. Set task_completed=true "
        "on the final successful update to signal that the agent is finished; do not rely on an empty "
        "tool-call response to indicate completion. A completion update must be the only call in its batch. "
        "At each reasoning cycle, update memory with all tool results available since the previous "
        "memory update before choosing the next action batch."
    ),
)
def update_memory_state(
    content: str | None = None,
    find: str | None = None,
    replace: str = "",
    regex: bool = False,
    append: str | None = None,
    task_completed: bool = False,
) -> ToolResult:
    state = current_agent_state.get()
    if state is None:
        raise ValueError("Working memory is unavailable outside an active agent run.")
    target = working_memory_path(state)
    if target is None:
        raise ValueError("The active agent has no initialized working-memory file.")
    plan = state.node_states.planner.plan
    if task_completed and plan is not None and len(plan.steps) != 1:
        raise ValueError("A completion memory update must run alone; review the other tool results first.")
    if not target.is_file():
        raise ValueError("The current working-memory file does not exist; the runtime must create it first.")
    original = "" if content is not None else target.read_text(encoding=UTF8_ENCODING)
    if content is not None:
        updated = content
        operation = "working_memory_replaced"
    elif append is not None:
        updated = original + append
        operation = "working_memory_appended"
    elif find is not None:
        if regex:
            updated, count = re.subn(find, replace, original)
            if count == 0:
                raise ValueError("The regular expression was not found in the working-memory file.")
        else:
            if find not in original:
                raise ValueError("The requested text was not found in the working-memory file.")
            updated = original.replace(find, replace, 1)
        operation = "working_memory_replaced_text"
    else:
        raise ValueError("Provide exactly one update mode: content, find, or append.")
    validate_working_memory(updated)
    target.write_text(updated, encoding=UTF8_ENCODING)
    result = _result(
        target,
        operation,
    )
    if task_completed and isinstance(result.result, dict):
        result.result["task_completed"] = True
    state.condense_reasoner_history(state.gather_tool_results())
    return result


__all__ = ["update_memory_state"]
