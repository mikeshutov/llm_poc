from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_SYNTHESIS_FOLLOW_UP = "What would you like to do next?"


class SynthesisToolSummary(BaseModel):
    used_tools: list[str] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    freshness: str = ""


class SynthesisResult(BaseModel):
    result: list[str]
    follow_up: str = ""
    clarifying_question: str = ""
    roundtrip_summary: str = ""
    tool_summary: SynthesisToolSummary = Field(default_factory=SynthesisToolSummary)

    @field_validator("result", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @model_validator(mode="after")
    def normalize_questions(self) -> "SynthesisResult":
        self.follow_up = self.follow_up.strip()
        self.clarifying_question = self.clarifying_question.strip()
        has_follow_up = bool(self.follow_up)
        has_clarifying_question = bool(self.clarifying_question)
        if has_follow_up and has_clarifying_question:
            self.follow_up = ""
        if not has_follow_up and not has_clarifying_question:
            self.follow_up = DEFAULT_SYNTHESIS_FOLLOW_UP
        return self
