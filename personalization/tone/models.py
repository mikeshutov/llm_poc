from typing import Literal
from uuid import UUID

from pydantic import BaseModel


ToneType = Literal["profile", "conversation"]
ToneVerbosity = Literal["concise", "balanced", "detailed"]
ToneFormality = Literal["casual", "neutral", "formal"]
ToneDirectness = Literal["low", "medium", "high"]
ToneHumor = Literal["none", "light", "frequent"]
ToneTechnicalDepth = Literal["low", "medium", "high"]


class TonePreferences(BaseModel):
    verbosity: ToneVerbosity | None = None
    formality: ToneFormality | None = None
    directness: ToneDirectness | None = None
    humor: ToneHumor | None = None
    technical_depth: ToneTechnicalDepth | None = None


class ToneRecord(TonePreferences):
    id: UUID | None = None
    user_id: str
    tone_type: ToneType = "profile"
    conversation_id: UUID | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_preferences(self) -> TonePreferences:
        return TonePreferences(
            verbosity=self.verbosity,
            formality=self.formality,
            directness=self.directness,
            humor=self.humor,
            technical_depth=self.technical_depth,
        )
