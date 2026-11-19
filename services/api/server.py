from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from common.config import FILES_DIR


FILES_ROOT = Path(FILES_DIR).resolve()
app = FastAPI(title="Application API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/files/{file_path:path}")
def serve_file(file_path: str) -> FileResponse:
    """Serve a managed file with a browser-appropriate content type."""
    candidate = (FILES_ROOT / file_path).resolve()
    try:
        candidate.relative_to(FILES_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    return FileResponse(
        candidate,
        media_type=media_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(candidate.name)}",
            "X-Content-Type-Options": "nosniff",
        },
    )
