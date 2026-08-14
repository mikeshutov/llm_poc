from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_SYNTHESIS_NEXT_QUESTION = "What would you like to do next?"


class SynthesisToolSummary(BaseModel):
    produced: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class SynthesisResultBlock(BaseModel):
    content: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    result: list[SynthesisResultBlock]
    next_question: str = ""
    roundtrip_summary: str = ""
    tool_summary: SynthesisToolSummary = Field(default_factory=SynthesisToolSummary)

    @model_validator(mode="before")
    @classmethod
    def merge_legacy_question_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("next_question"):
            return data
        clarifying_question = str(data.get("clarifying_question") or "").strip()
        follow_up = str(data.get("follow_up") or "").strip()
        merged = clarifying_question or follow_up
        if not merged:
            return data
        return {
            **data,
            "next_question": merged,
        }

    @field_validator("result", mode="before")
    @classmethod
    def validate_result_blocks(cls, v):
        if isinstance(v, str):
            raise ValueError("result must be a list of block objects with content and evidence_ids")
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    raise ValueError("result entries must be block objects, not strings")
        return v

    @model_validator(mode="after")
    def normalize_questions(self) -> "SynthesisResult":
        self.next_question = self.next_question.strip()
        if not self.next_question:
            self.next_question = DEFAULT_SYNTHESIS_NEXT_QUESTION
        self.result = [
            SynthesisResultBlock(
                content=block.content.strip(),
                evidence_ids=[evidence_id.strip() for evidence_id in block.evidence_ids if isinstance(evidence_id, str) and evidence_id.strip()],
            )
            for block in self.result
            if block.content.strip()
        ]
        return self
