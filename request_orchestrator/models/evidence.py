from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from common.html_text import html_to_plain_text, normalize_html_values


class EvidenceUrlType(StrEnum):
    WEBSITE = "website"
    YOUTUBE = "youtube"


class EvidenceUrl(BaseModel):
    url: str = ""
    url_type: EvidenceUrlType = EvidenceUrlType.WEBSITE


class CompactEvidenceView(BaseModel):
    evidence_id: UUID
    title: str = ""
    summary: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class HydratedEvidenceView(CompactEvidenceView):
    item_id: str = ""
    tool_name: str = ""
    urls: list[EvidenceUrl] = Field(default_factory=list)
    image_url: str = ""
    published_at: str = ""
    source: str = ""
    entity_type: str = ""
    location_name: str = ""


class EvaluatorEvidenceView(BaseModel):
    evidence_id: UUID
    summary: str = ""
    present_data: list[str] = Field(default_factory=list)


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
    llm_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    raw_payload: Any = None

    @field_validator("title", "summary", mode="before")
    @classmethod
    def normalize_llm_text(cls, value: object) -> object:
        return html_to_plain_text(value) if isinstance(value, str) else value

    @field_validator("llm_metadata", mode="before")
    @classmethod
    def normalize_llm_metadata(cls, value: object) -> object:
        return normalize_html_values(value)

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

    def compact_view(self) -> dict[str, Any]:
        return CompactEvidenceView(
            evidence_id=self.id,
            title=self.title,
            summary=self.summary,
            metadata=self.llm_metadata,
        ).model_dump(mode="json")

    def hydrated_view(self) -> dict[str, Any]:
        return HydratedEvidenceView(
            evidence_id=self.id,
            item_id=self.item_id,
            tool_name=self.tool_name,
            title=self.title,
            summary=self.summary,
            urls=self.urls,
            image_url=self.image_url,
            published_at=self.published_at,
            source=self.source,
            entity_type=self.entity_type,
            location_name=self.location_name,
            metadata=self.llm_metadata,
        ).model_dump(mode="json")

    def to_evaluator_view(self) -> dict[str, Any]:
        hydrated_data = {
            "title": self.title,
            "summary": self.summary,
            "urls": self.urls,
            "image_url": self.image_url,
            "published_at": self.published_at,
            "source": self.source,
            "entity_type": self.entity_type,
            "location_name": self.location_name,
        }
        return EvaluatorEvidenceView(
            evidence_id=self.id,
            summary=self.summary,
            present_data=[
                *[field_name for field_name, value in hydrated_data.items() if value],
                *sorted(self.llm_metadata),
            ],
        ).model_dump(mode="json")


class EvidenceBundle(BaseModel):
    evidence_by_id: dict[str, EvidenceView] = Field(default_factory=dict)
    evidence_views_by_tool_call_id: dict[UUID, list[EvidenceView]] = Field(default_factory=dict)


class ToolMetadata(BaseModel):
    """Shared, result-level metadata emitted by tool adapters."""

    model_config = ConfigDict(extra="forbid")
    retrieved_count: int | None = None
    reranked: bool | None = None
    product_source: str | list[str] | None = None
    search_type: str | None = None
    current_page: int | None = None
    page_size: int | None = None
    has_more: bool | None = None
    returned_count: int | None = None
    warnings: list[str] | None = None
    ruling_source: str | None = None


class ToolResult(BaseModel):
    tool_call_id: UUID | None = None
    plan_step_id: UUID | None = None
    tool_name: str = ""
    iteration: int | None = None
    result: Any = None
    tool_metadata: ToolMetadata = Field(default_factory=ToolMetadata)
    evidence: list[EvidenceView] = Field(default_factory=list)

    @classmethod
    def error(cls, error: str, **extra_result: Any) -> "ToolResult":
        payload: dict[str, Any] = {"error": error}
        if extra_result:
            payload.update(extra_result)
        return cls(result=payload, tool_metadata=ToolMetadata(), evidence=[])
