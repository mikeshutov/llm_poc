from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI


@dataclass
class FileParserState:
    raw_content: str
    file_name: str
    extracted: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    valid: bool = False
    llm: Any = field(
        default_factory=lambda: ChatOpenAI(model=os.getenv("AGENT_MODEL", "gpt-4.1"), temperature=0)
    )

    @classmethod
    def new(cls, raw_content: str, file_name: str) -> "FileParserState":
        return cls(raw_content=raw_content, file_name=file_name)
