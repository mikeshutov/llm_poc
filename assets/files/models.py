from __future__ import annotations

from dataclasses import dataclass

from assets.models import FileAsset

__all__ = [
    "FileAsset",
    "FileContentResult",
    "FileListResult",
    "FileOperationResult",
]


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
