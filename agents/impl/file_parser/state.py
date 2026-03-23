from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.agentstate.model import AgentState


@dataclass
class FileParserState(AgentState):
    raw_content: str = ""
    file_name: str = ""
    extracted: dict[str, Any] = field(default_factory=dict)
    valid: bool = False

    @classmethod
    def new(cls, raw_content: str, file_name: str, **kwargs: Any) -> "FileParserState":
        return cls(raw_content=raw_content, file_name=file_name, **kwargs)
