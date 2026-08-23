from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class EvidenceUrlType(StrEnum):
    WEBSITE = "website"
    YOUTUBE = "youtube"


class EvidenceUrl(BaseModel):
    url: str = ""
    url_type: EvidenceUrlType = EvidenceUrlType.WEBSITE


class EvidenceView(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tool_call_id: UUID | None = None
    hash: str = ""
    item_id: str = ""
    tool_name: str = ""
    title: str = ""
    summary: str = ""
    urls: list[EvidenceUrl] = Field(default_factory=list)
    image_url: str = ""
    published_at: str = ""
    source: str = ""
    entity_type: str = ""
    location_name: str = ""
    llm_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_payload: Any = None

    @property
    def url(self) -> str:
        for preferred_type in (EvidenceUrlType.WEBSITE, EvidenceUrlType.YOUTUBE):
            for entry in self.urls:
                cleaned_url = entry.url.strip()
                if entry.url_type == preferred_type and cleaned_url:
                    return cleaned_url
        for entry in self.urls:
            cleaned_url = entry.url.strip()
            if cleaned_url:
                return cleaned_url
        return ""

    def for_llm(self) -> dict[str, Any]:
        return {
            "evidence_id": str(self.id),
            "item_id": self.item_id,
            "title": self.title,
            "summary": self.summary,
            "metadata": dict(self.llm_metadata),
        }


class EvidenceBundle(BaseModel):
    evidence_by_id: dict[str, EvidenceView] = Field(default_factory=dict)
    evidence_views_by_tool_call_id: dict[UUID, list[EvidenceView]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_reloaded_evidence_views(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        def coerce_evidence(evidence: Any) -> Any:
            # Streamlit can retain tool modules across a source reload, leaving
            # them with an older EvidenceView class identity.
            return evidence.model_dump() if isinstance(evidence, BaseModel) else evidence

        normalized = dict(value)
        evidence_by_id = normalized.get("evidence_by_id")
        if isinstance(evidence_by_id, dict):
            normalized["evidence_by_id"] = {
                evidence_id: coerce_evidence(evidence)
                for evidence_id, evidence in evidence_by_id.items()
            }
        evidence_by_tool_call_id = normalized.get("evidence_views_by_tool_call_id")
        if isinstance(evidence_by_tool_call_id, dict):
            normalized["evidence_views_by_tool_call_id"] = {
                tool_call_id: [coerce_evidence(evidence) for evidence in evidence_views]
                if isinstance(evidence_views, list)
                else evidence_views
                for tool_call_id, evidence_views in evidence_by_tool_call_id.items()
            }
        return normalized


class ToolResult(BaseModel):
    tool_call_id: UUID | None = None
    plan_step_id: UUID | None = None
    tool_name: str = ""
    iteration: int | None = None
    result: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceView] = Field(default_factory=list)

    @classmethod
    def error(cls, error: str, **extra_result: Any) -> "ToolResult":
        payload: dict[str, Any] = {"error": error}
        if extra_result:
            payload.update(extra_result)
        return cls(result=payload, metadata={}, evidence=[])
