from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from assets.models import Asset, AssetChunk, AssetOrigin, AssetType, FileAsset

# File-specific result projections remain here; the canonical file entity is
# now assets.FileAsset.
__all__ = [
    "Asset",
    "AssetOrigin",
    "AssetType",
    "FileAsset",
    "File",
    "FileChunkResult",
    "AssetChunk",
    "FileContentResult",
    "FileListResult",
    "FileOperationResult",
]


File = FileAsset


@dataclass(frozen=True)
class FileChunkResult:
    file_id: UUID
    asset_type: str
    file_name: str
    file_path: str
    content: str


@dataclass(frozen=True)
class FileOperationResult:
    file: FileAsset
    message: str


@dataclass(frozen=True)
class FileContentResult:
    file: FileAsset
    content: str


@dataclass(frozen=True)
class FileListResult:
    files: list[FileAsset]
