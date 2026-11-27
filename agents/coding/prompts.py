from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from common.config import UTF8_ENCODING

@lru_cache(maxsize=1)
def build_coding_prompt() -> str:
    return Path(__file__).with_name("coding.md").read_text(encoding=UTF8_ENCODING).strip()


__all__ = [
    "build_coding_prompt",
]
