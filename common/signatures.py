from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


def build_signature(value: Any) -> str:
    """Return a stable SHA-256 signature for JSON-compatible domain data."""
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()
