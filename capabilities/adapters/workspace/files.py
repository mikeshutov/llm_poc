from __future__ import annotations

import fnmatch
import mimetypes
import re
import shutil
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent_runtime.context import get_current_conversation_id
from agent_runtime.context import get_current_user_id
from assets.files.storage import conversation_workspace_path, scoped_file_path
from assets.files.urls import static_file_url
from assets.files.models import FileListResult
from assets.files.file_types import FileType, TEXT_CONTENT_FILE_TYPES, WORKSPACE_FILE_TYPES
from assets.files.writer import normalize_file_format, write_file
from assets.models import AssetOrigin
from assets.repository.asset_repository import AssetRepository
from common.config import ANONYMOUS_OWNER
from capabilities.registry.constants import (
    TOOL_NAME_WORKSPACE_FILE_DELETE,
    TOOL_NAME_WORKSPACE_FILE_EDIT,
    TOOL_NAME_WORKSPACE_FILE_COMPLETE,
    TOOL_NAME_WORKSPACE_FILE_LIST,
    TOOL_NAME_WORKSPACE_FILE_READ,
    TOOL_NAME_WORKSPACE_FILE_WRITE,
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    TOOL_RESULT_TYPE_WORKSPACE_FILE,
)
from retrieval.evidence import EvidenceView, ToolResult


_MAX_READ_CHARS = 200_000


class WorkspaceFileNameArgs(BaseModel):
    file_name: str = Field(..., min_length=1, description="The name of a file in the root of the current conversation workspace.")


class WorkspaceWriteArgs(WorkspaceFileNameArgs):
    content: str = Field(default="", description="UTF-8 text content to write. Known document formats and arbitrary source-code extensions are supported.")
    overwrite: bool = Field(default=False, description="Whether to replace an existing file.")


class WorkspaceReadOptions(BaseModel):
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-based first line to return, inclusive.",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-based last line to return, inclusive.",
    )
    max_chars: int = Field(
        default=_MAX_READ_CHARS,
        ge=1,
        le=_MAX_READ_CHARS,
        description="Maximum number of characters to return after selecting the line range.",
    )


class WorkspaceReadArgs(WorkspaceFileNameArgs):
    options: WorkspaceReadOptions = Field(
        default_factory=WorkspaceReadOptions,
        description="Optional read range and output-size settings.",
    )


class WorkspaceEditArgs(WorkspaceFileNameArgs):
    find: str = Field(..., min_length=1, description="Exact text to replace.")
    replace: str = ""
    replace_all: bool = False


class WorkspaceListArgs(BaseModel):
    query: str | None = Field(
        default=None,
        description="Optional case-insensitive file name search or glob, such as '*.py' or 'report'.",
    )
    regex: str | None = Field(
        default=None,
        description="Optional regular expression matched against the relative file name.",
    )
    search_file_contents: bool = Field(
        default=False,
        description="When true, also search matching text-file contents using query or regex.",
    )
    recursive: bool = Field(default=True, description="Search nested directories when true.")


class WorkspacePromoteArgs(BaseModel):
    file_names: list[str] = Field(
        ...,
        min_length=1,
        description="One or more workspace file names to copy into managed files.",
    )


def _workspace_root() -> Path:
    user_id = (get_current_user_id() or ANONYMOUS_OWNER).strip()
    conversation_id = (get_current_conversation_id() or "").strip()
    if (
        not user_id
        or user_id in {".", ".."}
        or any(char in user_id for char in "/\\")
        or not conversation_id
        or conversation_id in {".", ".."}
        or any(char in conversation_id for char in "/\\")
    ):
        raise ValueError("A current conversation is required for workspace file access.")
    root = conversation_workspace_path(conversation_id, user_id=user_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_file_name(file_name: str) -> Path:
    if not file_name.strip() or file_name != file_name.strip():
        raise ValueError("A workspace file name is required.")
    candidate = Path(file_name)
    if candidate.is_absolute() or candidate.name != file_name or file_name in {".", ".."}:
        raise ValueError("Workspace file operations accept a file name, not a path.")
    root = _workspace_root()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Workspace path must remain inside the conversation workspace.")
    return resolved


def _relative(path: Path) -> str:
    return path.relative_to(_workspace_root()).as_posix()


def _read_searchable_text(path: Path) -> str | None:
    extension = path.suffix.lstrip(".").casefold()
    if extension in {file_type.value for file_type in TEXT_CONTENT_FILE_TYPES}:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
    mime_type = mimetypes.guess_type(path.name)[0] or ""
    if mime_type.startswith(("image/", "audio/", "video/")) or mime_type in {
        "application/pdf",
        "application/zip",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        return None
    try:
        raw = path.read_bytes()
        if b"\x00" in raw:
            return None
        return raw.decode("utf-8")
    except (OSError, UnicodeError):
        return None


def _result(path: Path, operation: str, *, content: str | None = None, items: list[str] | None = None) -> ToolResult:
    relative = _relative(path) if path != _workspace_root() else ""
    summary = f"Workspace file {operation}: {relative}" if relative else f"Workspace files {operation}."
    payload = {"path": relative, "workspace": str(path.parent if path != _workspace_root() else path), "operation": operation}
    if content is not None:
        payload["content"] = content
    if items is not None:
        payload["files"] = items
    evidence = EvidenceView(
        tool_name=operation,
        title=relative or "Workspace files",
        summary=summary,
        source=operation,
        entity_type=TOOL_RESULT_TYPE_WORKSPACE_FILE,
        llm_metadata={"path": relative, "workspace": str(_workspace_root()), "operation": operation},
    )
    return ToolResult(result=payload, evidence=[evidence])


@tool(TOOL_NAME_WORKSPACE_FILE_WRITE, args_schema=WorkspaceWriteArgs, description="Write UTF-8 text to a file in the current conversation workspace, including arbitrary source-code extensions.")
def workspace_file_write(file_name: str, content: str = "", overwrite: bool = False) -> ToolResult:
    target = _safe_file_name(file_name)
    if target.exists() and not overwrite:
        raise ValueError(f"Workspace file already exists: '{file_name}'. Set overwrite=true to replace it.")
    file_format = FileType.TXT
    if target.suffix:
        try:
            _, file_format = normalize_file_format(
                target.name,
                supported_types=WORKSPACE_FILE_TYPES,
            )
        except ValueError:
            # Workspace files can contain source code in any text-based
            # language; unknown extensions such as .go remain UTF-8 text.
            file_format = FileType.TXT
    write_file(target, content, file_format)
    return _result(target, "written")


@tool(TOOL_NAME_WORKSPACE_FILE_READ, args_schema=WorkspaceReadArgs, description="Read UTF-8 text from a file in the current conversation workspace.")
def workspace_file_read(
    file_name: str,
    options: WorkspaceReadOptions | None = None,
) -> ToolResult:
    target = _safe_file_name(file_name)
    if not target.is_file():
        raise ValueError(f"Workspace file does not exist: '{file_name}'.")
    read_options = options or WorkspaceReadOptions()
    if (
        read_options.start_line is not None
        and read_options.end_line is not None
        and read_options.end_line < read_options.start_line
    ):
        raise ValueError("end_line must be greater than or equal to start_line.")
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    start = (read_options.start_line or 1) - 1
    end = read_options.end_line
    content = "".join(lines[start:end])[: read_options.max_chars]
    return _result(target, "read", content=content)


@tool(TOOL_NAME_WORKSPACE_FILE_EDIT, args_schema=WorkspaceEditArgs, description="Apply an exact text replacement to a named file in the root of the current conversation workspace.")
def workspace_file_edit(file_name: str, find: str, replace: str = "", replace_all: bool = False) -> ToolResult:
    target = _safe_file_name(file_name)
    if not target.is_file():
        raise ValueError(f"Workspace file does not exist: '{file_name}'.")
    original = target.read_text(encoding="utf-8")
    occurrences = original.count(find)
    if occurrences == 0:
        raise ValueError("The requested text was not found in the workspace file.")
    if occurrences > 1 and not replace_all:
        raise ValueError("The requested text occurs multiple times; set replace_all=true to edit all matches.")
    target.write_text(original.replace(find, replace, -1 if replace_all else 1), encoding="utf-8")
    return _result(target, "edited")


@tool(TOOL_NAME_WORKSPACE_FILE_LIST, args_schema=WorkspaceListArgs, description="Search and list files in the current conversation workspace using a name query or regular expression, with optional text-content search.")
def workspace_file_list(
    query: str | None = None,
    regex: str | None = None,
    search_file_contents: bool = False,
    recursive: bool = True,
) -> ToolResult:
    directory = _workspace_root()
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    normalized_query = query.strip().casefold() if query else ""
    try:
        filename_pattern = re.compile(regex, re.IGNORECASE) if regex else None
    except re.error as exc:
        raise ValueError(f"Invalid file name regular expression: {exc}") from exc
    items = []
    for item in candidates:
        if not item.is_file():
            continue
        relative = _relative(item)
        normalized_relative = relative.casefold()
        name_matches = not normalized_query and filename_pattern is None
        if normalized_query and (
            normalized_query in normalized_relative
            or fnmatch.fnmatch(normalized_relative, normalized_query)
        ):
            name_matches = True
        if filename_pattern and filename_pattern.search(relative):
            name_matches = True
        if search_file_contents and (normalized_query or filename_pattern):
            text = _read_searchable_text(item)
            if text is not None:
                content_matches = bool(
                    normalized_query and normalized_query in text.casefold()
                ) or bool(filename_pattern and filename_pattern.search(text))
                name_matches = name_matches or content_matches
        if not name_matches:
            continue
        items.append(relative)
    items.sort()
    return _result(directory, "listed", items=items)


@tool(TOOL_NAME_WORKSPACE_FILE_DELETE, args_schema=WorkspaceFileNameArgs, description="Delete a named file from the current conversation workspace.")
def workspace_file_delete(file_name: str) -> ToolResult:
    target = _safe_file_name(file_name)
    if not target.is_file():
        raise ValueError(f"Workspace file does not exist: '{file_name}'.")
    target.unlink()
    return _result(target, "deleted")


@tool(
    TOOL_NAME_WORKSPACE_FILE_COMPLETE,
    args_schema=WorkspaceFileNameArgs,
    description="Mark a completed workspace file as the final artifact produced by the current agent.",
)
def workspace_file_complete(file_name: str) -> ToolResult:
    target = _safe_file_name(file_name)
    if not target.is_file():
        raise ValueError(f"Workspace file does not exist: '{file_name}'.")
    if not target.read_text(encoding="utf-8").strip():
        raise ValueError(f"Workspace file is empty: '{file_name}'.")
    return _result(target, "completed")


@tool(
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    args_schema=WorkspacePromoteArgs,
    description="Copy one or more completed workspace files into the user's managed files area, overwriting managed files with the same names, and return downloadable evidence for each.",
)
def workspace_file_promote(file_names: list[str]) -> ToolResult:
    sources = [_safe_file_name(file_name) for file_name in file_names]
    for file_name, source in zip(file_names, sources):
        if not source.is_file():
            raise ValueError(f"Workspace file does not exist: '{file_name}'.")

    managed_paths = [scoped_file_path(source.name, user_id=get_current_user_id()) for source in sources]
    duplicate_names = {path.name for path in managed_paths if managed_paths.count(path) > 1}
    if duplicate_names:
        raise ValueError(f"Multiple workspace files resolve to the same managed filename: {', '.join(sorted(duplicate_names))}.")

    repository = AssetRepository()
    user_id = get_current_user_id()
    files = []
    evidence = []
    for source, managed_path in zip(sources, managed_paths):
        managed_path.parent.mkdir(parents=True, exist_ok=True)
        existing_file = next(
            (
                file
                for file in repository.list_files(user_id=user_id, query=managed_path.name)
                if file.name == managed_path.name
            ),
            None,
        )
        shutil.copy2(source, managed_path)
        file = repository.create_file(
            asset_id=existing_file.id if existing_file else None,
            name=managed_path.name,
            storage_path=managed_path.as_posix(),
            mime_type=mimetypes.guess_type(managed_path.name)[0] or "application/octet-stream",
            user_id=user_id,
            origin=AssetOrigin.GENERATED,
            size_bytes=managed_path.stat().st_size,
        )
        files.append(file)
        file_url = static_file_url(file.storage_path)
        evidence.append(EvidenceView(
            item_id=str(file.id),
            tool_name=TOOL_NAME_WORKSPACE_FILE_PROMOTE,
            title=file.name,
            summary="Workspace file copied to managed files and is available for download.",
            urls=[{"url": file_url}] if file_url else [],
            source=TOOL_NAME_WORKSPACE_FILE_PROMOTE,
            entity_type="file",
            llm_metadata={
                "file_id": str(file.id),
                "filename": file.name,
                "mime_type": file.mime_type,
                "storage_path": file.storage_path,
            },
        ))
    return ToolResult(
        result=FileListResult(files=files),
        evidence=evidence,
    )


__all__ = [
    "workspace_file_write",
    "workspace_file_read",
    "workspace_file_edit",
    "workspace_file_list",
    "workspace_file_delete",
    "workspace_file_complete",
    "workspace_file_promote",
]
