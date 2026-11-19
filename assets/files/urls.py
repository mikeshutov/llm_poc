from pathlib import PurePosixPath
from urllib.parse import quote
import os

from common.config import FILES_DIR


_STATIC_FILES_PATH = PurePosixPath(FILES_DIR.replace("\\", "/"))
_STREAMLIT_STATIC_PREFIX = PurePosixPath("app")
STATIC_FILES_URL_PREFIX = (
    f"/{_STREAMLIT_STATIC_PREFIX.as_posix()}/{_STATIC_FILES_PATH.as_posix()}/"
)
_FILES_ROUTE = "/api/files"


def artifact_file_url(file_path: str) -> str:
    """Return a browser URL served by the application API."""
    normalized_path = file_path.replace("\\", "/").strip()
    while normalized_path.startswith("./"):
        normalized_path = normalized_path[2:]

    path = PurePosixPath(normalized_path)
    if any(part in {".", ".."} for part in path.parts):
        return ""
    if path.parts[: len(_STATIC_FILES_PATH.parts)] != _STATIC_FILES_PATH.parts:
        return ""

    relative_path = PurePosixPath(*path.parts[len(_STATIC_FILES_PATH.parts):])
    if not relative_path.parts:
        return ""
    base_url = (os.getenv("API_SERVER_URL") or os.getenv("ARTIFACT_SERVER_URL", "")).strip().rstrip("/")
    return quote(f"{base_url}{_FILES_ROUTE}/{relative_path.as_posix()}", safe="/:")


def static_file_url(file_path: str) -> str:
    """Backward-compatible name for the browser-facing file URL."""
    return artifact_file_url(file_path)


__all__ = [
    "STATIC_FILES_URL_PREFIX",
    "artifact_file_url",
    "static_file_url",
]
