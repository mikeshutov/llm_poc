from __future__ import annotations

from pydantic import BaseModel, Field


class AvailableAgent(BaseModel):
    agent: str
    description: str = ""
    tools: list[str] = Field(default_factory=list)


class AvailableTool(BaseModel):
    """A concrete tool selected as relevant for the current request."""

    name: str
    description: str = ""


class CapabilityDiscoveryResult(BaseModel):
    agents: list[AvailableAgent] = Field(default_factory=list)
    tools: list[AvailableTool] = Field(default_factory=list)
