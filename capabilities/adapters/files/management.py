from __future__ import annotations

import mimetypes
import os
from pathlib import Path
import re
from uuid import UUID

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.config import CHUNK_ENCODING, FILES_DIR
from assets.models import FileAsset
from assets.files.models import FileContentResult, FileListResult, FileOperationResult
from assets.models import AssetOrigin
from assets.repository.asset_chunk_repository import AssetChunkRepository
from assets.repository.asset_repository import AssetRepository
from assets.files.storage import scoped_file_path
from assets.files.urls import static_file_url
from agent_runtime.context import get_current_user_id
from retrieval.evidence import ToolResult
from capabilities.registry.constants import (
    TOOL_NAME_FILE_CREATE,
    TOOL_NAME_FILE_DELETE,
    TOOL_NAME_FILE_EDIT,
    TOOL_NAME_FILE_LIST,
    TOOL_NAME_FILE_MOVE,
    TOOL_NAME_FILE_READ,
    TOOL_RESULT_TYPE_FILE,
)
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView


_FILES_ROOT = Path(FILES_DIR).resolve()
_MAX_READ_CHARS = 200_000


class FileIdArgs(BaseModel):
    file_id: str = Field(..., description="The UUID of the file.")


class FileCreateArgs(BaseModel):
    file_name: str = Field(..., min_length=1, description="The new file name, relative to the managed files directory.")
    content: str = Field(default="", description="Text content for the new file or document.")
    format: str | None = Field(default=None, description="Output format such as txt, md, docx, pdf, or svg. If omitted, infer it from file_name.")
    title: str | None = Field(default=None, description="Optional document title, used for structured formats such as docx.")


class FileReadArgs(FileIdArgs):
    max_chars: int = Field(default=_MAX_READ_CHARS, ge=1, le=_MAX_READ_CHARS)


class FileEditArgs(FileIdArgs):
    find: str = Field(..., min_length=1, description="The exact text to replace.")
    replace: str = Field(default="", description="Replacement text.")
    replace_all: bool = False


class FileMoveArgs(FileIdArgs):
    destination: str = Field(..., min_length=1, description="The new relative file name or path.")


class FileListArgs(BaseModel):
    query: str | None = Field(default=None, description="Optional file-name or path filter.")


def _parse_id(file_id: str) -> UUID:
    try:
        return UUID(file_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid file_id '{file_id}'.") from exc


def _safe_path(file_path: str) -> Path:
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    if resolved != _FILES_ROOT and _FILES_ROOT not in resolved.parents:
        raise ValueError("File path must remain inside the managed files directory.")
    return resolved


def _stored_path(path: Path) -> str:
    return os.path.join(FILES_DIR, path.relative_to(_FILES_ROOT).as_posix())


def _chunks(content: str) -> list[str]:
    if not content:
        return []
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=CHUNK_ENCODING,
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_text(content)


def _index(file_id: UUID, content: str) -> None:
    chunks = AssetChunkRepository()
    chunks.delete_chunks(file_id)
    chunks.replace_content(file_id, "file", _chunks(content))


def _file_evidence(file: FileAsset, *, tool_name: str, summary: str) -> EvidenceView:
    file_url = static_file_url(file.storage_path)
    return EvidenceView(
        item_id=str(file.id),
        tool_name=tool_name,
        title=file.name,
        summary=summary,
        urls=[EvidenceUrl(url=file_url, url_type=EvidenceUrlType.WEBSITE)] if file_url else [],
        source=tool_name,
        entity_type=TOOL_RESULT_TYPE_FILE,
        llm_metadata={
            "file_id": str(file.id),
            "filename": file.name,
            "mime_type": file.mime_type,
            "storage_path": file.storage_path,
            "size_bytes": file.size_bytes,
        },
        raw_payload=file,
    )


def _file_result(file: FileAsset, message: str, *, tool_name: str) -> ToolResult:
    return ToolResult(
        result=FileOperationResult(file=file, message=message),
        evidence=[_file_evidence(file, tool_name=tool_name, summary=message)],
    )


def _file_created_result(file: FileAsset) -> ToolResult:
    """Return the created file plus a canonical downloadable evidence view."""
    evidence = _file_evidence(
        file,
        tool_name=TOOL_NAME_FILE_CREATE,
        summary="File created and available for download.",
    )
    return ToolResult(
        result=FileOperationResult(file=file, message="File created."),
        evidence=[evidence],
    )


def _normalize_format(file_name: str, file_format: str | None) -> tuple[str, str]:
    has_explicit_format = bool(file_format and file_format.strip())
    requested = (file_format or Path(file_name).suffix.lstrip(".") or "txt").strip().lower()
    requested = requested.lstrip(".")
    aliases = {
        "text": "txt",
        "plain text": "txt",
        "text/plain": "txt",
        "markdown": "md",
        "text/markdown": "md",
        "word": "docx",
        "doc": "docx",
        "docx": "docx",
        "wordprocessingml": "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "pdf": "pdf",
        "application/pdf": "pdf",
        "svg": "svg",
        "image/svg+xml": "svg",
    }
    normalized = aliases.get(requested)
    if normalized is None:
        # Accept descriptive values emitted by agents while retaining a small,
        # explicit set of supported output formats.
        if "word" in requested or "wordprocessingml" in requested or "openxmlformats" in requested:
            normalized = "docx"
        elif "pdf" in requested:
            normalized = "pdf"
        elif "markdown" in requested:
            normalized = "md"
        elif "text" in requested:
            normalized = "txt"
        else:
            normalized = requested
    if normalized not in {"txt", "md", "docx", "pdf", "svg"}:
        raise ValueError("Unsupported file format. Supported formats are txt, md, docx, pdf, and svg.")
    path = Path(file_name)
    if not path.suffix:
        file_name = f"{file_name}.{normalized}"
    elif has_explicit_format and path.suffix.lower().lstrip(".") != normalized:
        file_name = str(path.with_suffix(f".{normalized}"))
    elif not has_explicit_format and path.suffix.lower() == ".doc":
        # The writer emits DOCX, so never store it with the legacy .doc suffix.
        file_name = str(path.with_suffix(".docx"))
    return file_name, normalized


def _document_paragraphs(content: str) -> list[str]:
    """Normalize user text into deterministic paragraphs without rewriting it."""
    normalized = (content or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalized:
        return [""]
    return [
        " ".join(line.strip() for line in block.splitlines()).strip()
        for block in re.split(r"\n\s*\n", normalized)
        if block.strip()
    ]


def _write_document(path: Path, content: str, file_format: str, title: str | None) -> None:
    if file_format in {"txt", "md", "svg"}:
        path.write_text(content, encoding="utf-8")
        return
    if file_format == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.utils import simpleSplit
            from reportlab.pdfgen.canvas import Canvas
        except ImportError as exc:
            raise ValueError("PDF creation requires the reportlab package.") from exc
        canvas = Canvas(str(path), pagesize=letter)
        width, height = letter
        text = canvas.beginText(54, height - 54)
        text.setFont("Helvetica", 12)
        if title and title.strip():
            text.setFont("Helvetica-Bold", 16)
            text.textLine(title.strip())
            text.textLine("")
            text.setFont("Helvetica", 12)
        paragraphs = _document_paragraphs(content)
        for paragraph_index, paragraph in enumerate(paragraphs):
            lines = simpleSplit(paragraph, "Helvetica", 12, width - 108) or [""]
            for line in lines:
                if text.getY() < 54:
                    canvas.drawText(text)
                    canvas.showPage()
                    text = canvas.beginText(54, height - 54)
                    text.setFont("Helvetica", 12)
                text.textLine(line)
            if paragraph_index < len(paragraphs) - 1:
                if text.getY() < 54:
                    canvas.drawText(text)
                    canvas.showPage()
                    text = canvas.beginText(54, height - 54)
                    text.setFont("Helvetica", 12)
                text.textLine("")
        canvas.drawText(text)
        canvas.save()
        return
    document = Document()
    if title and title.strip():
        document.add_heading(title.strip(), level=1)
    for paragraph in _document_paragraphs(content):
        document.add_paragraph(paragraph)
    document.save(path)


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError as exc:
            raise ValueError("PDF reading requires the pdfplumber package.") from exc
    return path.read_text(encoding="utf-8")


@tool(TOOL_NAME_FILE_CREATE, args_schema=FileCreateArgs, description="Create a new managed text, Word, PDF, or SVG file. Supports txt, md, docx, pdf, and svg formats.")
def file_create(
    file_name: str,
    content: str = "",
    format: str | None = None,
    title: str | None = None,
) -> ToolResult:
    try:
        file_name, file_format = _normalize_format(file_name, format)
        target = _safe_path(scoped_file_path(file_name, user_id=get_current_user_id()))
        if target.exists():
            raise ValueError(f"A file already exists at '{file_name}'.")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_document(target, content, file_format, title)
        file = AssetRepository().create_file(
            name=target.name,
            storage_path=_stored_path(target),
            mime_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if file_format == "docx"
                else "image/svg+xml"
                if file_format == "svg"
                else mimetypes.guess_type(target.name)[0] or "text/plain"
            ),
            user_id=get_current_user_id(),
            origin=AssetOrigin.GENERATED,
        )
        _index(file.id, content)
        return _file_created_result(file)
    except (OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


@tool(TOOL_NAME_FILE_READ, args_schema=FileReadArgs, description="Retrieve the complete UTF-8 text content of a managed file when full content is needed.")
def file_read(file_id: str, max_chars: int = _MAX_READ_CHARS) -> ToolResult:
    row = AssetRepository().get_file(_parse_id(file_id), user_id=get_current_user_id())
    if not row:
        raise ValueError(f"No file found with id '{file_id}'.")
    file = row
    try:
        content = _read_document(_safe_path(file.storage_path))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Unable to read file '{file.name}'.") from exc
    return ToolResult(
        result=FileContentResult(file=file, content=content[:max_chars]),
        evidence=[_file_evidence(file, tool_name=TOOL_NAME_FILE_READ, summary="File content retrieved.")],
    )


@tool(TOOL_NAME_FILE_EDIT, args_schema=FileEditArgs, description="Apply an exact targeted text replacement to a managed UTF-8 file.")
def file_edit(file_id: str, find: str, replace: str = "", replace_all: bool = False) -> ToolResult:
    row = AssetRepository().get_file(_parse_id(file_id), user_id=get_current_user_id())
    if not row:
        raise ValueError(f"No file found with id '{file_id}'.")
    file = row
    path = _safe_path(file.storage_path)
    try:
        original = _read_document(path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Unable to edit file '{file.name}'.") from exc
    occurrences = original.count(find)
    if occurrences == 0:
        raise ValueError("The requested text was not found in the file.")
    if occurrences > 1 and not replace_all:
        raise ValueError("The requested text occurs multiple times; set replace_all=true to edit all matches.")
    updated = original.replace(find, replace, -1 if replace_all else 1)
    _write_document(path, updated, "docx" if path.suffix.lower() == ".docx" else path.suffix.lower().lstrip(".") or "txt", None)
    _index(file.id, updated)
    return _file_result(
        file,
        f"File edited; replaced {occurrences if replace_all else 1} occurrence(s).",
        tool_name=TOOL_NAME_FILE_EDIT,
    )


@tool(TOOL_NAME_FILE_DELETE, args_schema=FileIdArgs, description="Delete a managed file and its indexed content.")
def file_delete(file_id: str) -> ToolResult:
    parsed_id = _parse_id(file_id)
    row = AssetRepository().get_file(parsed_id, user_id=get_current_user_id())
    if not row:
        raise ValueError(f"No file found with id '{file_id}'.")
    file = row
    path = _safe_path(file.storage_path)
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        raise ValueError(f"Unable to delete file '{file.name}'.") from exc
    repository = AssetRepository()
    deleted = repository.delete_file(parsed_id, user_id=get_current_user_id())
    if not deleted:
        raise ValueError(f"No file found with id '{file_id}'.")
    return _file_result(row, "File deleted.", tool_name=TOOL_NAME_FILE_DELETE)


@tool(TOOL_NAME_FILE_LIST, args_schema=FileListArgs, description="List managed files visible to the current user.")
def file_list(query: str | None = None) -> ToolResult:
    files = AssetRepository().list_files(user_id=get_current_user_id(), query=query)
    return ToolResult(
        result=FileListResult(files=files),
        evidence=[
            _file_evidence(file, tool_name=TOOL_NAME_FILE_LIST, summary="Managed file available.")
            for file in files
        ],
    )


@tool(TOOL_NAME_FILE_MOVE, args_schema=FileMoveArgs, description="Rename or relocate a managed file within the managed files directory.")
def file_move(file_id: str, destination: str) -> ToolResult:
    parsed_id = _parse_id(file_id)
    row = AssetRepository().get_file(parsed_id, user_id=get_current_user_id())
    if not row:
        raise ValueError(f"No file found with id '{file_id}'.")
    file = row
    source = _safe_path(file.storage_path)
    target = _safe_path(os.path.join(FILES_DIR, destination))
    if target.exists():
        raise ValueError(f"A file already exists at '{destination}'.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.rename(target)
    except OSError as exc:
        raise ValueError(f"Unable to move file '{file.name}'.") from exc
    repository = AssetRepository()
    updated = repository.update_file_path(
        parsed_id,
        storage_path=_stored_path(target),
        name=target.name,
        user_id=get_current_user_id(),
    )
    if not updated:
        raise ValueError(f"Unable to update metadata for file '{file.name}'.")
    return _file_result(
        repository.get_file(parsed_id, user_id=get_current_user_id()),
        "File moved.",
        tool_name=TOOL_NAME_FILE_MOVE,
    )
