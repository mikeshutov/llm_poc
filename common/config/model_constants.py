from __future__ import annotations

import os

CHUNK_ENCODING = "cl100k_base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

AVAILABLE_CHAT_MODELS = [
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-4o-mini",
    "o3",
]
