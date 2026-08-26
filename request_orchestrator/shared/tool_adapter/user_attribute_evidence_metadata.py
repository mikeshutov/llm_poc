from typing import Literal

from pydantic import BaseModel, Field


class UserAttributeEvidenceMetadata(BaseModel):
    operation: Literal["created", "updated"]
    group_key: str | None = None
    attribute_values: list[str] = Field(default_factory=list)
