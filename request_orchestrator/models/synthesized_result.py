from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class SynthesisResultBlock(BaseModel):
    content: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    result: list[SynthesisResultBlock]
    next_question: str = ""
    roundtrip_summary: str = ""

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
    def normalize_result(self) -> "SynthesisResult":
        self.next_question = self.next_question.strip()
        self.result = [
            SynthesisResultBlock(
                content=block.content.strip(),
                evidence_ids=[evidence_id.strip() for evidence_id in block.evidence_ids if isinstance(evidence_id, str) and evidence_id.strip()],
            )
            for block in self.result
            if block.content.strip()
        ]
        return self
