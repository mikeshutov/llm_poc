from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personalization.profile.repository.repo_factory import get_user_profile_repo
from request_orchestrator.shared.runtime_context import get_current_user_id
from request_orchestrator.shared.tool_adapter.profile.result_models import UserProfileUpdateResult


class SetUserFirstNameArgs(BaseModel):
    first_name: str = Field(
        ...,
        description="The user's first name when it is directly known from the current request.",
    )


@tool(
    "set_user_first_name",
    args_schema=SetUserFirstNameArgs,
    description="Set the user's first name when it is missing and explicitly provided.",
)
def set_user_first_name(first_name: str) -> UserProfileUpdateResult:
    user_id = get_current_user_id()
    if user_id is None or not user_id.strip():
        raise ValueError("A current user_id is required to update the user profile.")

    profile = get_user_profile_repo().get_profile(user_id)
    if profile is None:
        profile = get_user_profile_repo().ensure_profile(user_id)

    if (profile.first_name or "").strip():
        raise ValueError("first_name is already set for this user profile and cannot be updated with this tool.")

    resolved_first_name = first_name.strip()
    if not resolved_first_name:
        raise ValueError("first_name must not be empty.")

    updated_profile = get_user_profile_repo().update_profile(
        user_id=user_id,
        first_name=resolved_first_name,
        last_name=profile.last_name,
        display_name=profile.display_name,
        email=profile.email,
    )
    if updated_profile is None:
        raise ValueError(f"Could not update profile for user_id={user_id}")

    return UserProfileUpdateResult(
        user_id=updated_profile.user_id,
        first_name=updated_profile.first_name,
        last_name=updated_profile.last_name,
        display_name=updated_profile.display_name,
    )
