from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetType(StrEnum):
    FILE = "file"
    HTTP_PAGE = "http_page"


class AssetContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class AssetOrigin(StrEnum):
    UPLOADED = "uploaded"
    GENERATED = "generated"
    DERIVED = "derived"


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    user_id: str | None = None
    asset_type: str
    origin: AssetOrigin
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime | None = None

    @field_validator("name")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("asset name is required")
        return value


class FileAsset(Asset):
    asset_type: Literal[AssetType.FILE] = AssetType.FILE
    storage_path: str
    mime_type: str
    size_bytes: int | None = Field(default=None, ge=0)
    checksum: str | None = None


class HttpPage(Asset):
    asset_type: Literal[AssetType.HTTP_PAGE] = AssetType.HTTP_PAGE
    url: str
    title: str = ""
    content: str = ""
    fetched_at: datetime | None = None


class AssetChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: UUID
    asset_type: str
    chunk_index: int = Field(ge=0)
    content: str
    embedding: list[float]


class AssetChunkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: UUID
    asset_type: str
    asset_name: str
    content: str
    storage_path: str | None = None
    mime_type: str | None = None
    url: str | None = None
    title: str | None = None


class AssetReference(BaseModel):
    """Reference to a canonical asset associated with a job result."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    user_id: str | None = None
    asset_type: str
    asset_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @field_validator("asset_type", "asset_id")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("asset_type and asset_id are required")
        return value
