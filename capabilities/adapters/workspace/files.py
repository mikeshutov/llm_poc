from __future__ import annotations

import fnmatch
import mimetypes
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent_runtime.context import get_current_conversation_id
from agent_runtime.context import get_current_user_id
from assets.files.storage import scoped_file_path
from assets.files.urls import static_file_url
from assets.files.models import FileListResult
from assets.models import AssetOrigin
from assets.repository.asset_repository import AssetRepository
from common.config import FILES_DIR
from capabilities.registry.constants import (
    TOOL_NAME_WORKSPACE_FILE_DELETE,
    TOOL_NAME_WORKSPACE_FILE_EDIT,
    TOOL_NAME_WORKSPACE_FILE_LIST,
    TOOL_NAME_WORKSPACE_FILE_READ,
    TOOL_NAME_WORKSPACE_FILE_WRITE,
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    TOOL_RESULT_TYPE_WORKSPACE_FILE,
)
from retrieval.evidence import EvidenceView, ToolResult


_MAX_READ_CHARS = 200_000


class WorkspacePathArgs(BaseModel):
    path: str = Field(..., min_length=1, description="A relative path inside the conversation workspace files directory.")


class WorkspaceWriteArgs(WorkspacePathArgs):
    content: str = Field(default="", description="UTF-8 text content to write.")
    overwrite: bool = Field(default=False, description="Whether to replace an existing file.")


class WorkspaceReadArgs(WorkspacePathArgs):
    max_chars: int = Field(default=_MAX_READ_CHARS, ge=1, le=_MAX_READ_CHARS)


class WorkspaceEditArgs(WorkspacePathArgs):
    find: str = Field(..., min_length=1, description="Exact text to replace.")
    replace: str = ""
    replace_all: bool = False


class WorkspaceListArgs(BaseModel):
    path: str = Field(default="", description="Optional relative directory inside the workspace files directory.")
    query: str | None = Field(
        default=None,
        description="Optional case-insensitive file path search or glob, such as '*.py' or 'report'.",
    )
    recursive: bool = Field(default=True, description="Search nested directories when true.")


class WorkspacePromoteArgs(BaseModel):
    paths: list[str] = Field(
        ...,
        min_length=1,
        description="One or more relative workspace file paths to move into managed files.",
    )


def _workspace_root() -> Path:
    user_id = (get_current_user_id() or "anonymous").strip()
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
    root = (Path(FILES_DIR) / user_id / conversation_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_path(path: str, *, allow_empty: bool = False) -> Path:
    if not path.strip() and not allow_empty:
        raise ValueError("A workspace file path is required.")
    relative = Path(path)
    if relative.is_absolute() or relative == Path(".") and path.strip() not in {"", "."}:
        raise ValueError("Workspace paths must be relative.")
    root = _workspace_root()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Workspace path must remain inside the conversation workspace.")
    return resolved


def _relative(path: Path) -> str:
    return path.relative_to(_workspace_root()).as_posix()


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


@tool(TOOL_NAME_WORKSPACE_FILE_WRITE, args_schema=WorkspaceWriteArgs, description="Write UTF-8 text to a file in the current conversation workspace.")
def workspace_file_write(path: str, content: str = "", overwrite: bool = False) -> ToolResult:
    target = _safe_path(path)
    if target.exists() and not overwrite:
        raise ValueError(f"Workspace file already exists: '{path}'. Set overwrite=true to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return _result(target, "written")


@tool(TOOL_NAME_WORKSPACE_FILE_READ, args_schema=WorkspaceReadArgs, description="Read UTF-8 text from a file in the current conversation workspace.")
def workspace_file_read(path: str, max_chars: int = _MAX_READ_CHARS) -> ToolResult:
    target = _safe_path(path)
    if not target.is_file():
        raise ValueError(f"Workspace file does not exist: '{path}'.")
    return _result(target, "read", content=target.read_text(encoding="utf-8")[:max_chars])


@tool(TOOL_NAME_WORKSPACE_FILE_EDIT, args_schema=WorkspaceEditArgs, description="Apply an exact text replacement to a file in the current conversation workspace.")
def workspace_file_edit(path: str, find: str, replace: str = "", replace_all: bool = False) -> ToolResult:
    target = _safe_path(path)
    if not target.is_file():
        raise ValueError(f"Workspace file does not exist: '{path}'.")
    original = target.read_text(encoding="utf-8")
    occurrences = original.count(find)
    if occurrences == 0:
        raise ValueError("The requested text was not found in the workspace file.")
    if occurrences > 1 and not replace_all:
        raise ValueError("The requested text occurs multiple times; set replace_all=true to edit all matches.")
    target.write_text(original.replace(find, replace, -1 if replace_all else 1), encoding="utf-8")
    return _result(target, "edited")


@tool(TOOL_NAME_WORKSPACE_FILE_LIST, args_schema=WorkspaceListArgs, description="Search and list files in the current conversation workspace using an optional path, query, and recursive flag.")
def workspace_file_list(path: str = "", query: str | None = None, recursive: bool = True) -> ToolResult:
    directory = _safe_path(path, allow_empty=True)
    if not directory.is_dir():
        raise ValueError(f"Workspace directory does not exist: '{path}'.")
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    normalized_query = query.strip().casefold() if query else ""
    items = []
    for item in candidates:
        if not item.is_file():
            continue
        relative = _relative(item)
        normalized_relative = relative.casefold()
        if normalized_query and not (
            normalized_query in normalized_relative
            or fnmatch.fnmatch(normalized_relative, normalized_query)
        ):
            continue
        items.append(relative)
    items.sort()
    return _result(directory, "listed", items=items)


@tool(TOOL_NAME_WORKSPACE_FILE_DELETE, args_schema=WorkspacePathArgs, description="Delete a file from the current conversation workspace.")
def workspace_file_delete(path: str) -> ToolResult:
    target = _safe_path(path)
    if not target.is_file():
        raise ValueError(f"Workspace file does not exist: '{path}'.")
    target.unlink()
    return _result(target, "deleted")


@tool(
    TOOL_NAME_WORKSPACE_FILE_PROMOTE,
    args_schema=WorkspacePromoteArgs,
    description="Move one or more completed workspace files into the user's managed files area and return downloadable evidence for each.",
)
def workspace_file_promote(paths: list[str]) -> ToolResult:
    sources = [_safe_path(path) for path in paths]
    for path, source in zip(paths, sources):
        if not source.is_file():
            raise ValueError(f"Workspace file does not exist: '{path}'.")

    managed_paths = [scoped_file_path(source.name, user_id=get_current_user_id()) for source in sources]
    duplicate_names = {path.name for path in managed_paths if managed_paths.count(path) > 1}
    if duplicate_names:
        raise ValueError(f"Multiple workspace files resolve to the same managed filename: {', '.join(sorted(duplicate_names))}.")
    existing = [path.name for path in managed_paths if path.exists()]
    if existing:
        raise ValueError(f"Managed files already exist: {', '.join(existing)}.")

    repository = AssetRepository()
    files = []
    evidence = []
    for source, managed_path in zip(sources, managed_paths):
        managed_path.parent.mkdir(parents=True, exist_ok=True)
        source.rename(managed_path)
        file = repository.create_file(
            name=managed_path.name,
            storage_path=managed_path.as_posix(),
            mime_type=mimetypes.guess_type(managed_path.name)[0] or "application/octet-stream",
            user_id=get_current_user_id(),
            origin=AssetOrigin.GENERATED,
        )
        files.append(file)
        file_url = static_file_url(file.storage_path)
        evidence.append(EvidenceView(
            item_id=str(file.id),
            tool_name=TOOL_NAME_WORKSPACE_FILE_PROMOTE,
            title=file.name,
            summary="Workspace file moved to managed files and is available for download.",
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


__all__ = ["workspace_file_write", "workspace_file_read", "workspace_file_edit", "workspace_file_list", "workspace_file_delete", "workspace_file_promote"]
