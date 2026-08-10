from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personalization.profile.repository.repo_factory import get_user_profile_repo
from request_orchestrator.shared.runtime_context import get_current_user_id


class SetUserLastNameArgs(BaseModel):
    last_name: str = Field(
        ...,
        description="The user's last name when it is directly known from the current request.",
    )


@tool(
    "set_user_last_name",
    args_schema=SetUserLastNameArgs,
    description="Set the user's last name when it is missing and explicitly provided.",
)
def set_user_last_name(last_name: str) -> dict[str, str | None]:
    user_id = get_current_user_id()
    if user_id is None or not user_id.strip():
        raise ValueError("A current user_id is required to update the user profile.")

    profile = get_user_profile_repo().get_profile(user_id)
    if profile is None:
        profile = get_user_profile_repo().ensure_profile(user_id)

    updated_profile = get_user_profile_repo().update_profile(
        user_id=user_id,
        first_name=profile.first_name,
        last_name=last_name.strip(),
        display_name=profile.display_name,
        email=profile.email,
    )
    if updated_profile is None:
        raise ValueError(f"Could not update profile for user_id={user_id}")

    return {
        "user_id": updated_profile.user_id,
        "first_name": updated_profile.first_name,
        "last_name": updated_profile.last_name,
        "display_name": updated_profile.display_name,
    }
