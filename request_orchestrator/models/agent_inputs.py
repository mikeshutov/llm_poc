from __future__ import annotations

from dataclasses import dataclass, field

from request_orchestrator.shared.discovery.models import CapabilityDiscoveryResult


@dataclass
class AgentInputs:
    task: str = ""
    request_task: str = ""
    tool_names: list[str] = field(default_factory=list)
    tool_category_names: list[str] = field(default_factory=list)
    discovered_capabilities: CapabilityDiscoveryResult | None = None

    @classmethod
    def new(
        cls,
        *,
        task: str = "",
        request_task: str | None = None,
        tool_names: list[str] | None = None,
        tool_category_names: list[str] | None = None,
        discovered_capabilities: CapabilityDiscoveryResult | None = None,
    ) -> "AgentInputs":
        return cls(
            task=task.strip(),
            request_task=(task if request_task is None else request_task).strip(),
            tool_names=[] if tool_names is None else [
                name.strip()
                for name in tool_names
                if isinstance(name, str) and name.strip()
            ],
            tool_category_names=[] if tool_category_names is None else [
                category.strip()
                for category in tool_category_names
                if isinstance(category, str) and category.strip()
            ],
            discovered_capabilities=discovered_capabilities,
        )
