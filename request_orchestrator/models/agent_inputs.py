from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentInputs:
    task: str = ""
    tool_category_names: list[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        *,
        task: str = "",
        tool_category_names: list[str] | None = None,
    ) -> "AgentInputs":
        return cls(
            task=task.strip(),
            tool_category_names=[] if tool_category_names is None else [
                category.strip()
                for category in tool_category_names
                if isinstance(category, str) and category.strip()
            ],
        )
