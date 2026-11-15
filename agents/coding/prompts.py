from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def build_coding_prompt() -> str:
    return Path(__file__).with_name("coding.md").read_text(encoding="utf-8").strip()


__all__ = [
    "build_coding_prompt",
]
