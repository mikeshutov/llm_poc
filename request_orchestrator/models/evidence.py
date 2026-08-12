from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceUrl(BaseModel):
    url: str = ""
    url_type: str = "website"


class HydratedEvidence(BaseModel):
    evidence_id: str = ""
    step_id: str = ""
    item_id: str = ""
    tool_name: str = ""
    title: str = ""
    summary: str = ""
    url: str = ""
    urls: list[EvidenceUrl] = Field(default_factory=list)
    image_url: str = ""
    published_at: str = ""
    source: str = ""
    entity_type: str = ""
    location_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_payload: Any = None


class EvidenceView(BaseModel):
    evidence_id: str = ""
    item_id: str = ""
    title: str = ""
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    hydrated_evidence_by_id: dict[str, HydratedEvidence] = Field(default_factory=dict)
    evidence_views_by_step_id: dict[str, list[EvidenceView]] = Field(default_factory=dict)
