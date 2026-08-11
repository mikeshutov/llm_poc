from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from personalization.profile.models import UserProfile
from personalization.profile.repository.repo_factory import get_user_profile_repo
from personalization.tone.models import (
    ToneDirectness,
    ToneFormality,
    ToneHumor,
    TonePreferences,
    ToneTechnicalDepth,
    ToneVerbosity,
)
from request_orchestrator.shared.runtime_context import get_current_user_id

MIN_TONE_UPDATE_CONFIDENCE = 0.9


class UpdateUserToneArgs(BaseModel):
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence between 0 and 1 that the user expressed a durable tone preference.",
    )
    verbosity: ToneVerbosity | None = Field(
        default=None,
        description="How concise or detailed the user generally prefers communication to be, or how they themselves tend to communicate. Allowed values: concise, balanced, detailed.",
    )
    formality: ToneFormality | None = Field(
        default=None,
        description="How casual or formal the user generally prefers communication to be, or how they themselves tend to communicate. Allowed values: casual, neutral, formal.",
    )
    directness: ToneDirectness | None = Field(
        default=None,
        description="How direct or indirect the user generally prefers communication to be, or how they themselves tend to communicate. Allowed values: low, medium, high.",
    )
    humor: ToneHumor | None = Field(
        default=None,
        description="How much humor the user generally prefers in communication, or how much humor they themselves tend to use. Allowed values: none, light, frequent.",
    )
    technical_depth: ToneTechnicalDepth | None = Field(
        default=None,
        description="How technically deep the user generally prefers communication to be, or how technically deep they themselves tend to communicate. Allowed values: low, medium, high.",
    )

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> UpdateUserToneArgs:
        if all(
            value is None
            for value in (
                self.verbosity,
                self.formality,
                self.directness,
                self.humor,
                self.technical_depth,
            )
        ):
            raise ValueError("At least one tone field must be provided.")
        return self


class UpdateUserToneResult(BaseModel):
    user_id: str | None
    applied: bool
    status: str
    reason: str = ""
    confidence: float
    minimum_confidence: float
    tone: TonePreferences | None = None


def _merge_tone_preferences(
    profile: UserProfile,
    *,
    verbosity: ToneVerbosity | None,
    formality: ToneFormality | None,
    directness: ToneDirectness | None,
    humor: ToneHumor | None,
    technical_depth: ToneTechnicalDepth | None,
) -> TonePreferences:
    existing = profile.tone or TonePreferences()
    return TonePreferences(
        verbosity=existing.verbosity if verbosity is None else verbosity,
        formality=existing.formality if formality is None else formality,
        directness=existing.directness if directness is None else directness,
        humor=existing.humor if humor is None else humor,
        technical_depth=existing.technical_depth if technical_depth is None else technical_depth,
    )


def _tone_preferences_equal(left: TonePreferences | None, right: TonePreferences | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return left.model_dump() == right.model_dump()


@tool(
    "update_user_tone",
    args_schema=UpdateUserToneArgs,
    description="Update durable user tone preferences when the user clearly expresses how they want responses to sound.",
)
def update_user_tone(
    confidence: float,
    verbosity: ToneVerbosity | None = None,
    formality: ToneFormality | None = None,
    directness: ToneDirectness | None = None,
    humor: ToneHumor | None = None,
    technical_depth: ToneTechnicalDepth | None = None,
) -> UpdateUserToneResult:
    user_id = get_current_user_id()
    if user_id is None or not user_id.strip():
        raise ValueError("A current user_id is required to update the user profile.")

    profile_repo = get_user_profile_repo()
    profile = profile_repo.get_profile(user_id)
    if profile is None:
        profile = profile_repo.ensure_profile(user_id)

    if confidence < MIN_TONE_UPDATE_CONFIDENCE:
        return UpdateUserToneResult(
            user_id=profile.user_id,
            applied=False,
            status="rejected",
            reason=(
                f"Tone update rejected because confidence {confidence:.2f} is below "
                f"the minimum threshold of {MIN_TONE_UPDATE_CONFIDENCE:.2f}."
            ),
            confidence=confidence,
            minimum_confidence=MIN_TONE_UPDATE_CONFIDENCE,
            tone=profile.tone,
        )

    merged_tone = _merge_tone_preferences(
        profile,
        verbosity=verbosity,
        formality=formality,
        directness=directness,
        humor=humor,
        technical_depth=technical_depth,
    )
    if _tone_preferences_equal(profile.tone, merged_tone):
        return UpdateUserToneResult(
            user_id=profile.user_id,
            applied=False,
            status="unchanged",
            reason="Tone update skipped because the requested values do not change the stored tone preference.",
            confidence=confidence,
            minimum_confidence=MIN_TONE_UPDATE_CONFIDENCE,
            tone=profile.tone,
        )

    updated_profile = profile_repo.update_profile(
        user_id=user_id,
        first_name=profile.first_name,
        last_name=profile.last_name,
        display_name=profile.display_name,
        email=profile.email,
        tone=merged_tone,
    )
    if updated_profile is None:
        raise ValueError(f"Could not update profile for user_id={user_id}")

    return UpdateUserToneResult(
        user_id=updated_profile.user_id,
        applied=True,
        status="updated",
        confidence=confidence,
        minimum_confidence=MIN_TONE_UPDATE_CONFIDENCE,
        tone=updated_profile.tone,
    )
