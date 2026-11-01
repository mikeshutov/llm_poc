from __future__ import annotations

from pydantic import BaseModel, Field


class AvailableAgent(BaseModel):
    agent: str
    description: str = ""
    tools: list[str] = Field(default_factory=list)

    def as_available_tool(self) -> "AvailableTool":
        """Project an agent into the common LLM-facing capability shape."""
        return AvailableTool(
            name=self.agent,
            title="Agent capability",
            category="agents",
            description=self.description,
        )


class AvailableTool(BaseModel):
    """A capability exposed to the LLM for the current request."""

    name: str
    title: str = ""
    category: str = ""
    description: str = ""

    @classmethod
    def from_tool(cls, tool: object, *, category: str = "") -> "AvailableTool":
        """Create the prompt-facing descriptor for an executable tool."""
        if isinstance(tool, cls):
            if category and not tool.category:
                return tool.model_copy(update={"category": category})
            return tool
        name = str(getattr(tool, "name", "")).strip()
        return cls(
            name=name,
            title=str(getattr(tool, "title", "") or name),
            category=category,
            description=str(getattr(tool, "description", "") or ""),
        )


class CapabilityDiscoveryResult(BaseModel):
    agents: list[AvailableAgent] = Field(default_factory=list)
    tools: list[AvailableTool] = Field(default_factory=list)

    def as_available_tools(self, *, include_agents: bool = True) -> list[AvailableTool]:
        """Return all discovered capabilities in the prompt-facing shape."""
        capabilities = list(self.tools)
        if include_agents:
            capabilities.extend(agent.as_available_tool() for agent in self.agents)
        return capabilities
