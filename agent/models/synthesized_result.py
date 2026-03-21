from __future__ import annotations

from pydantic import BaseModel, field_validator


class SynthesisResult(BaseModel):
    result: list[str]
    follow_up: str
    clarifying_question: str = ""

    @field_validator("result", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v