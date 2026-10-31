from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.config import CHUNK_ENCODING, FILES_DIR
from assets.models import FileAsset
from assets.files.models import FileContentResult, FileListResult, FileOperationResult
from assets.models import AssetOrigin
from assets.repository.asset_chunk_repository import AssetChunkRepository
from assets.repository.asset_repository import AssetRepository
from agent_runtime.context import get_current_conversation_id, get_current_user_id
from retrieval.evidence import ToolResult
from capabilities.registry.constants import (
    TOOL_NAME_FILE_CREATE,
    TOOL_NAME_FILE_DELETE,
    TOOL_NAME_FILE_EDIT,
    TOOL_NAME_FILE_LIST,
    TOOL_NAME_FILE_MOVE,
    TOOL_NAME_FILE_READ,
)


_FILES_ROOT = Path(FILES_DIR).resolve()
_MAX_READ_CHARS = 200_000


class FileIdArgs(BaseModel):
    file_id: str = Field(..., description="The UUID of the file.")


class FileCreateArgs(BaseModel):
    file_name: str = Field(..., min_length=1, description="The new file name, relative to the managed files directory.")
    content: str = Field(default="", description="UTF-8 text content for the new file.")


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


def _file_result(file: FileAsset, message: str) -> ToolResult:
    return ToolResult(result=FileOperationResult(file=file, message=message))


@tool(TOOL_NAME_FILE_CREATE, args_schema=FileCreateArgs, description="Create a new UTF-8 text file in the managed files directory.")
def file_create(file_name: str, content: str = "") -> ToolResult:
    try:
        target = _safe_path(os.path.join(FILES_DIR, file_name))
        if target.exists():
            raise ValueError(f"A file already exists at '{file_name}'.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        file = AssetRepository().create_file(
            name=target.name,
            storage_path=_stored_path(target),
            mime_type=mimetypes.guess_type(target.name)[0] or "text/plain",
            user_id=get_current_user_id(),
            origin=AssetOrigin.GENERATED,
            conversation_id=(
                UUID(conversation_id)
                if (conversation_id := get_current_conversation_id())
                else None
            ),
        )
        _index(file.id, content)
        return _file_result(file, "File created.")
    except (OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


@tool(TOOL_NAME_FILE_READ, args_schema=FileReadArgs, description="Retrieve the complete UTF-8 text content of a managed file when full content is needed.")
def file_read(file_id: str, max_chars: int = _MAX_READ_CHARS) -> ToolResult:
    row = AssetRepository().get_file(_parse_id(file_id), user_id=get_current_user_id())
    if not row:
        raise ValueError(f"No file found with id '{file_id}'.")
    file = row
    try:
        content = _safe_path(file.storage_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read file '{file.name}'.") from exc
    return ToolResult(result=FileContentResult(file=file, content=content[:max_chars]))


@tool(TOOL_NAME_FILE_EDIT, args_schema=FileEditArgs, description="Apply an exact targeted text replacement to a managed UTF-8 file.")
def file_edit(file_id: str, find: str, replace: str = "", replace_all: bool = False) -> ToolResult:
    row = AssetRepository().get_file(_parse_id(file_id), user_id=get_current_user_id())
    if not row:
        raise ValueError(f"No file found with id '{file_id}'.")
    file = row
    path = _safe_path(file.storage_path)
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to edit file '{file.name}'.") from exc
    occurrences = original.count(find)
    if occurrences == 0:
        raise ValueError("The requested text was not found in the file.")
    if occurrences > 1 and not replace_all:
        raise ValueError("The requested text occurs multiple times; set replace_all=true to edit all matches.")
    updated = original.replace(find, replace, -1 if replace_all else 1)
    path.write_text(updated, encoding="utf-8")
    _index(file.id, updated)
    return _file_result(file, f"File edited; replaced {occurrences if replace_all else 1} occurrence(s).")


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
    return _file_result(row, "File deleted.")


@tool(TOOL_NAME_FILE_LIST, args_schema=FileListArgs, description="List managed files visible to the current user.")
def file_list(query: str | None = None) -> ToolResult:
    return ToolResult(result=FileListResult(files=AssetRepository().list_files(user_id=get_current_user_id(), query=query)))


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
    return _file_result(repository.get_file(parsed_id, user_id=get_current_user_id()), "File moved.")
