from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent_runtime.context import get_current_user_id
from capabilities.adapters.profile.result_models import UserProfileUpdateResult
from capabilities.registry.constants import TOOL_NAME_UPDATE_USER_NAMES, TOOL_RESULT_TYPE_PROFILE
from personalization.profile.repository.repo_factory import get_user_profile_repo
from retrieval.evidence import EvidenceView, ToolResult


class UpdateUserNamesArgs(BaseModel):
    display_name: str = Field(
        default="",
        description="The user's preferred nickname or short display name; leave empty to keep it unchanged.",
    )
    first_name: str = Field(
        default="",
        description="The user's first name; leave empty to keep it unchanged.",
    )
    last_name: str = Field(
        default="",
        description="The user's last name; leave empty to keep it unchanged.",
    )


@tool(
    TOOL_NAME_UPDATE_USER_NAMES,
    args_schema=UpdateUserNamesArgs,
    description=(
        "Update any supplied user name fields. Empty fields are ignored and leave the "
        "existing value unchanged."
    ),
)
def update_user_names(
    display_name: str = "",
    first_name: str = "",
    last_name: str = "",
) -> ToolResult:
    user_id = get_current_user_id()
    if user_id is None or not user_id.strip():
        raise ValueError("A current user_id is required to update the user profile.")

    profile_repo = get_user_profile_repo()
    profile = profile_repo.get_profile(user_id) or profile_repo.ensure_profile(user_id)
    resolved_display_name = display_name.strip()
    resolved_first_name = first_name.strip()
    resolved_last_name = last_name.strip()
    updated_profile = profile_repo.update_profile(
        user_id=user_id,
        first_name=resolved_first_name or profile.first_name,
        last_name=resolved_last_name or profile.last_name,
        display_name=resolved_display_name or profile.display_name,
        email=profile.email,
    ) if any((resolved_display_name, resolved_first_name, resolved_last_name)) else profile
    if updated_profile is None:
        raise ValueError(f"Could not update profile for user_id={user_id}")

    result = UserProfileUpdateResult(
        user_id=updated_profile.user_id,
        first_name=updated_profile.first_name,
        last_name=updated_profile.last_name,
        display_name=updated_profile.display_name,
    )
    evidence_view = EvidenceView(
        item_id=str(result.user_id or ""),
        tool_name=TOOL_NAME_UPDATE_USER_NAMES,
        title="User Profile",
        summary="Updated supplied user name fields; empty fields were left unchanged.",
        source=TOOL_NAME_UPDATE_USER_NAMES,
        entity_type=TOOL_RESULT_TYPE_PROFILE,
        llm_metadata={},
        raw_payload=result,
    )
    return ToolResult(result=result, evidence=[evidence_view])
