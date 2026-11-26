from __future__ import annotations

import os

CHUNK_ENCODING = "cl100k_base"
UTF8_ENCODING = "utf-8"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
