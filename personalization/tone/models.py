from enum import Enum

from pydantic import BaseModel, Field


class ToneLabel(str, Enum):
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    PROFESSIONAL = "professional"


class Tone(BaseModel):
    label: ToneLabel
    score: float = Field(ge=0.0, le=1.0)
    override: bool = False
