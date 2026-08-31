from pathlib import PurePosixPath
from urllib.parse import quote

from common.config import FILES_DIR


_STATIC_FILES_PATH = PurePosixPath(FILES_DIR.replace("\\", "/"))
_STREAMLIT_STATIC_PREFIX = PurePosixPath("app")


def static_file_url(file_path: str) -> str:
    """Return a relative static URL only for files in the configured served directory."""
    normalized_path = file_path.replace("\\", "/").strip()
    while normalized_path.startswith("./"):
        normalized_path = normalized_path[2:]

    path = PurePosixPath(normalized_path)
    if any(part in {".", ".."} for part in path.parts):
        return ""
    if path.parts[: len(_STATIC_FILES_PATH.parts)] != _STATIC_FILES_PATH.parts:
        return ""

    return quote((_STREAMLIT_STATIC_PREFIX / path).as_posix(), safe="/")
