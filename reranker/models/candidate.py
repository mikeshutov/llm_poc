from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.html_text import html_to_plain_text, normalize_html_values
from reranker.models.candidate_content import CandidateContent


class Candidate(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    candidate_type: str | None = None
    title: str | None = None
    content: CandidateContent = Field(default_factory=CandidateContent)
    attributes: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    embedding: list[float] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        return html_to_plain_text(value) if isinstance(value, str) else value

    @field_validator("attributes", "metadata", mode="before")
    @classmethod
    def normalize_prompt_metadata(cls, value: object) -> object:
        return normalize_html_values(value)
