from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from common.config import (
    ANONYMOUS_OWNER,
    ARTIFACTS_DIR,
    FILES_DIR,
    WORKSPACE_DIR,
    WORKSPACES_DIR,
)

WORKING_MEMORY_DIRECTORY = "memory"


def _owner_segment(user_id: str | None) -> str:
    owner = user_id.strip() if isinstance(user_id, str) else ""
    if not owner:
        return ANONYMOUS_OWNER
    if owner in {".", ".."} or any(separator in owner for separator in ("/", "\\")):
        raise ValueError("Invalid user id for file storage.")
    return owner


def scoped_file_name(file_name: str, *, user_id: str | None) -> str:
    """Return a safe name; anonymous files never use the requested stem."""
    requested = Path(file_name)
    if not requested.name or requested.name in {".", ".."}:
        raise ValueError("A valid file name is required.")
    if requested.name != file_name or requested.name in {".", ".."}:
        raise ValueError("File name must not contain a directory path.")
    if user_id and user_id.strip():
        return requested.name
    return f"{uuid4().hex}{requested.suffix}"


def scoped_file_path(file_name: str, *, user_id: str | None) -> Path:
    return user_artifacts_path(file_name, user_id=user_id)


def user_artifacts_path(file_name: str, *, user_id: str | None) -> Path:
    return (
        Path(FILES_DIR)
        / _owner_segment(user_id)
        / ARTIFACTS_DIR
        / scoped_file_name(file_name, user_id=user_id)
    )


def conversation_workspace_path(conversation_id: str, *, user_id: str | None) -> Path:
    conversation = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if (
        not conversation
        or conversation in {".", ".."}
        or any(separator in conversation for separator in ("/", "\\"))
    ):
        raise ValueError("Invalid conversation id for file storage.")
    return (
        Path(FILES_DIR)
        / _owner_segment(user_id)
        / WORKSPACES_DIR
        / conversation
        / WORKSPACE_DIR
    )


def conversation_working_memory_path(
    conversation_id: str,
    file_name: str,
    *,
    user_id: str | None,
) -> Path:
    """Resolve a runtime memory file inside the conversation memory directory."""
    candidate = Path(file_name)
    if (
        not file_name
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("Invalid working-memory file name.")

    workspace = conversation_workspace_path(conversation_id, user_id=user_id)
    if candidate.parts[0] == WORKING_MEMORY_DIRECTORY:
        return workspace.joinpath(*candidate.parts)

    return workspace / WORKING_MEMORY_DIRECTORY / candidate


__all__ = [
    "conversation_workspace_path",
    "conversation_working_memory_path",
    "scoped_file_name",
    "scoped_file_path",
    "user_artifacts_path",
]
