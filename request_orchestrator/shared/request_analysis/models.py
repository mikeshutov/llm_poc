from __future__ import annotations

from pydantic import BaseModel, Field


class AvailableAgentToolCategory(BaseModel):
    name: str
    description: str = ""


class AvailableAgent(BaseModel):
    agent: str
    description: str = ""
    tool_categories: list[AvailableAgentToolCategory] = Field(default_factory=list)
