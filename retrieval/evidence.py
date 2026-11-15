from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from common.html_text import html_to_plain_text, normalize_html_values
from retrieval.reranker_models.reranker import Candidate


class EvidenceUrlType(StrEnum):
    WEBSITE = "website"
    YOUTUBE = "youtube"


class EvidenceUrl(BaseModel):
    url: str = ""
    url_type: EvidenceUrlType = EvidenceUrlType.WEBSITE


class EvidenceView(BaseModel):
    """Stable envelope carrying a tool-defined, typed data payload."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str = "generic"
    source: str = ""
    summary: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data", "metadata", mode="before")
    @classmethod
    def normalize_mapping(cls, value: object) -> dict[str, Any]:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        normalized = normalize_html_values(dict(value or {}))
        return TypeAdapter(dict[str, Any]).dump_python(normalized, mode="json")

    @field_validator("id", "type", "source", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return str(value or "")

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: object) -> object:
        return html_to_plain_text(value) if isinstance(value, str) else value

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: object) -> dict[str, Any]:
        return normalize_html_values(value or {})

    def to_candidate(self) -> Candidate:
        fields = {**self.data, **self.metadata}
        return Candidate(
            id=str(self.data.get("item_id") or self.id),
            fields={
                name: value for name, value in fields.items()
                if value is not None and value != "" and value != [] and value != {}
            },
        )

EVIDENCE_TYPES: dict[str, type[BaseModel]] = {}
EvidenceModel = TypeVar("EvidenceModel", bound=BaseModel)


def register_evidence_type(type_name: str, model: type[EvidenceModel]) -> type[EvidenceModel]:
    EVIDENCE_TYPES[type_name] = model
    return model


def reconstruct_evidence_data(type_name: str, data: Mapping[str, Any]) -> Any:
    model = EVIDENCE_TYPES.get(type_name)
    return model.model_validate(data) if model is not None else dict(data)


def from_evidence(view: EvidenceView) -> BaseModel:
    """Reconstruct a registered typed payload from an evidence envelope."""
    model_type = EVIDENCE_TYPES[view.type]
    return model_type.model_validate(view.data)


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


class ToolOutcomeStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    FAILED = "failed"


class ToolOutcome(BaseModel):
    status: ToolOutcomeStatus
    reason: str | None = None
    missing_inputs: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool_call_id: UUID | None = None
    plan_step_id: UUID | None = None
    tool_name: str = ""
    iteration: int | None = None
    hypothesis: str = ""
    call_metadata: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    tool_metadata: ToolMetadata = Field(default_factory=ToolMetadata)
    evidence: list[EvidenceView] = Field(default_factory=list)
    state: ToolOutcome | None = None

    @classmethod
    def error(cls, error: str, **extra_result: Any) -> "ToolResult":
        payload: dict[str, Any] = {"error": error}
        if extra_result:
            payload.update(extra_result)
        return cls(result=payload, tool_metadata=ToolMetadata(), evidence=[])
