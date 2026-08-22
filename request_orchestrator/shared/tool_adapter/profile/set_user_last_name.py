from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personalization.profile.repository.repo_factory import get_user_profile_repo
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from request_orchestrator.shared.tool_adapter.profile.result_models import UserProfileUpdateResult
from tool.constants import TOOL_NAME_SET_USER_LAST_NAME
from tool.constants import TOOL_RESULT_TYPE_PROFILE


class SetUserLastNameArgs(BaseModel):
    last_name: str = Field(
        ...,
        description="The user's last name when it is directly known from the current request.",
    )


def _tool_result(result: UserProfileUpdateResult) -> ToolResult:
    hydrated = HydratedEvidence(
        item_id=str(result.user_id or ""),
        tool_name=TOOL_NAME_SET_USER_LAST_NAME,
        title="User Profile",
        summary=f"Last name set to {(result.last_name or '').strip() or 'unknown'}.",
        source=TOOL_NAME_SET_USER_LAST_NAME,
        entity_type=TOOL_RESULT_TYPE_PROFILE,
        metadata={
            "user_id": result.user_id,
            "first_name": result.first_name,
            "last_name": result.last_name,
            "display_name": result.display_name,
        },
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence_views=[
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        ],
        hydrated_evidence=[hydrated],
    )


@tool(
    TOOL_NAME_SET_USER_LAST_NAME,
    args_schema=SetUserLastNameArgs,
    description="Set the user's last name when it is missing and explicitly provided.",
)
def set_user_last_name(last_name: str) -> ToolResult:
    user_id = get_current_user_id()
    if user_id is None or not user_id.strip():
        raise ValueError("A current user_id is required to update the user profile.")

    profile = get_user_profile_repo().get_profile(user_id)
    if profile is None:
        profile = get_user_profile_repo().ensure_profile(user_id)

    if (profile.last_name or "").strip():
        raise ValueError("last_name is already set for this user profile and cannot be updated with this tool.")

    resolved_last_name = last_name.strip()
    if not resolved_last_name:
        raise ValueError("last_name must not be empty.")

    updated_profile = get_user_profile_repo().update_profile(
        user_id=user_id,
        first_name=profile.first_name,
        last_name=resolved_last_name,
        display_name=profile.display_name,
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
