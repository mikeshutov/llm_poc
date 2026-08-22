from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceUrlType(StrEnum):
    WEBSITE = "website"
    YOUTUBE = "youtube"


class EvidenceUrl(BaseModel):
    url: str = ""
    url_type: EvidenceUrlType = EvidenceUrlType.WEBSITE


class HydratedEvidence(BaseModel):
    evidence_id: str = ""
    step_id: str = ""
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_object: Any = None
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


class EvidenceView(BaseModel):
    evidence_id: str = ""
    item_id: str = ""
    title: str = ""
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_object: Any = None


class EvidenceBundle(BaseModel):
    hydrated_evidence_by_id: dict[str, HydratedEvidence] = Field(default_factory=dict)
    evidence_views_by_step_id: dict[str, list[EvidenceView]] = Field(default_factory=dict)


class ToolResult(BaseModel):
    step_id: str = ""
    tool_name: str = ""
    iteration: int | None = None
    result: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_views: list[EvidenceView] = Field(default_factory=list)
    hydrated_evidence: list[HydratedEvidence] = Field(default_factory=list)

    @classmethod
    def error(cls, error: str, **extra_result: Any) -> "ToolResult":
        payload: dict[str, Any] = {"error": error}
        if extra_result:
            payload.update(extra_result)
        return cls(result=payload, metadata={}, evidence_views=[], hydrated_evidence=[])
