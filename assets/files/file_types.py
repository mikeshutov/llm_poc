from __future__ import annotations

from enum import StrEnum


class FileType(StrEnum):
    TXT = "txt"
    MARKDOWN = "md"
    DOCX = "docx"
    PDF = "pdf"
    SVG = "svg"
    CSV = "csv"
    JSON = "json"
    PYTHON = "py"
    JAVASCRIPT = "js"
    TYPESCRIPT = "ts"
    HTML = "html"
    CSS = "css"
    YAML = "yaml"
    YML = "yml"
    XML = "xml"
    SQL = "sql"
    SHELL = "sh"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    WEBP = "webp"


TEXT_FILE_TYPES = frozenset({
    FileType.PDF,
    FileType.TXT,
    FileType.DOCX,
})

IMAGE_FILE_TYPES = frozenset({
    FileType.PNG,
    FileType.JPG,
    FileType.JPEG,
    FileType.WEBP,
})


DOCUMENT_FILE_TYPES = frozenset({
    FileType.TXT,
    FileType.MARKDOWN,
    FileType.DOCX,
    FileType.PDF,
    FileType.SVG,
})

WORKSPACE_FILE_TYPES = frozenset(FileType)


def describe_file_types(file_types: frozenset[FileType]) -> str:
    return ", ".join(file_type.value for file_type in sorted(file_types, key=lambda item: item.value))


DOCUMENT_FILE_TYPES_DESCRIPTION = describe_file_types(DOCUMENT_FILE_TYPES)
WORKSPACE_FILE_TYPES_DESCRIPTION = describe_file_types(WORKSPACE_FILE_TYPES)


__all__ = [
    "DOCUMENT_FILE_TYPES",
    "DOCUMENT_FILE_TYPES_DESCRIPTION",
    "FileType",
    "IMAGE_FILE_TYPES",
    "TEXT_FILE_TYPES",
    "WORKSPACE_FILE_TYPES",
    "WORKSPACE_FILE_TYPES_DESCRIPTION",
    "describe_file_types",
]
