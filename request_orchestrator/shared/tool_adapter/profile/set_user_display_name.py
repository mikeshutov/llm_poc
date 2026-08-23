from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personalization.profile.repository.repo_factory import get_user_profile_repo
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from request_orchestrator.shared.tool_adapter.profile.result_models import UserProfileUpdateResult
from tool.constants import TOOL_NAME_SET_USER_DISPLAY_NAME
from tool.constants import TOOL_RESULT_TYPE_PROFILE


class SetUserDisplayNameArgs(BaseModel):
    display_name: str = Field(
        ...,
        description="The user's preferred nickname or short display name.",
    )


def _tool_result(result: UserProfileUpdateResult) -> ToolResult:
    hydrated = EvidenceView(
        item_id=str(result.user_id or ""),
        tool_name=TOOL_NAME_SET_USER_DISPLAY_NAME,
        title="User Profile",
        summary=f"Display name set to {(result.display_name or '').strip() or 'unknown'}.",
        source=TOOL_NAME_SET_USER_DISPLAY_NAME,
        entity_type=TOOL_RESULT_TYPE_PROFILE,
        llm_metadata={},
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[hydrated],
    )


@tool(
    TOOL_NAME_SET_USER_DISPLAY_NAME,
    args_schema=SetUserDisplayNameArgs,
    description="Set the user's nickname or preferred display name.",
)
def set_user_display_name(display_name: str) -> ToolResult:
    user_id = get_current_user_id()
    if user_id is None or not user_id.strip():
        raise ValueError("A current user_id is required to update the user profile.")

    profile = get_user_profile_repo().get_profile(user_id)
    if profile is None:
        profile = get_user_profile_repo().ensure_profile(user_id)

    updated_profile = get_user_profile_repo().update_profile(
        user_id=user_id,
        first_name=profile.first_name,
        last_name=profile.last_name,
        display_name=display_name.strip(),
        email=profile.email,
    )
    if updated_profile is None:
        raise ValueError(f"Could not update profile for user_id={user_id}")

    return _tool_result(
        UserProfileUpdateResult(
            user_id=updated_profile.user_id,
            first_name=updated_profile.first_name,
            last_name=updated_profile.last_name,
            display_name=updated_profile.display_name,
        )
    )
