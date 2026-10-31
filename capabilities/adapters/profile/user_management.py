from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_runtime.models.agent_result import ExecutionStatus
from capabilities.adapters.profile.result_models import UserManagementResult

if TYPE_CHECKING:
    from agent_runtime.models.agent_state import AgentState
    from agent_runtime.models.main_state import MainState


class UserManagementArgs(BaseModel):
    task: str = Field(..., min_length=1, description="Describe the profile-management task to perform.")


_CURRENT_MAIN_STATE: ContextVar[Any] = ContextVar("user_management_main_state", default=None)
_CURRENT_AGENT_STATE: ContextVar[Any] = ContextVar("user_management_agent_state", default=None)


def bind_user_management_state(state: MainState):
    return _CURRENT_MAIN_STATE.set(state)


def reset_user_management_state(token) -> None:
    _CURRENT_MAIN_STATE.reset(token)


def bind_user_management_agent_state(state: AgentState):
    return _CURRENT_AGENT_STATE.set(state)


def reset_user_management_agent_state(token) -> None:
    _CURRENT_AGENT_STATE.reset(token)


def _run_user_management(task: str) -> UserManagementResult:
    from agent_runtime.models.agent_profile import PROFILE_MANAGEMENT_AGENT_NAME
    from agent_runtime.models.agent_inputs import AgentInputs

    main_state = _CURRENT_MAIN_STATE.get()
    current_agent_state = _CURRENT_AGENT_STATE.get()
    if main_state is None:
        return UserManagementResult(error="User management is unavailable outside an active request.")

    delegated = main_state.agent_states.get(PROFILE_MANAGEMENT_AGENT_NAME)
    if delegated is None and current_agent_state is not None and current_agent_state.available_agent_states:
        delegated = current_agent_state.available_agent_states.get(PROFILE_MANAGEMENT_AGENT_NAME)
    if delegated is None:
        return UserManagementResult(error="The profile-management agent is unavailable for this request.")

    delegated.inputs = AgentInputs.new(
        task=task.strip(),
        request_task=delegated.inputs.request_task,
    )
    try:
        from agents.registry import agent_registry

        runner = agent_registry.get(delegated.agent_profile)
        completed = runner(delegated)
    except Exception as exc:
        return UserManagementResult(
            agent=PROFILE_MANAGEMENT_AGENT_NAME,
            status=ExecutionStatus.FAILED,
            error=str(exc),
        )

    tool_results = completed.gather_tool_results()
    return UserManagementResult(
        agent=PROFILE_MANAGEMENT_AGENT_NAME,
        status=completed.result.execution_status,
        answer_status=completed.result.answer_status,
        tool_names=completed.gather_used_tools(),
        tool_call_ids=[str(value) for value in completed.result.tool_call_ids],
        result_count=len(tool_results),
    )


USER_MANAGEMENT_TOOL = StructuredTool.from_function(
    func=_run_user_management,
    name="user_management",
    description=(
        "Manage durable user profile information such as names, preferences, tone, "
        "and user attributes. Describe the requested profile-management task."
    ),
    args_schema=UserManagementArgs,
)

