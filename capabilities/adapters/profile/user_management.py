from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_runtime.context import get_current_agent_state
from retrieval.evidence import ToolResult


class UserManagementArgs(BaseModel):
    task: str = Field(..., min_length=1, description="Describe the profile-management task to perform.")


def _run_user_management(task: str) -> ToolResult:
    from agent_runtime.execution.tool_executor import execute_agent_tool
    from agent_runtime.models.agent_profile import AgentName

    state = get_current_agent_state()
    if state is None:
        return ToolResult.error("User management is unavailable outside an active request.")
    return execute_agent_tool(
        state, AgentName.PROFILE_MANAGEMENT, {"task": task}, tool_name="user_management",
    )


USER_MANAGEMENT_TOOL = StructuredTool.from_function(
    func=_run_user_management,
    name="user_management",
    description=(
        "Manage durable user profile information such as names, preferences, tone, "
        "and user attributes. Describe the requested profile-management task."
    ),
    args_schema=UserManagementArgs,
    metadata={"agent_name": "profile_management"},
)

