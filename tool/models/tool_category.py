from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCategory:
    tools: list
    description: str
    rules: list[str] = field(default_factory=list)
    result_rules: list[str] = field(default_factory=list)
