from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictStr, confloat


class AttributeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr | None = None
    text: Annotated[StrictStr, Field(min_length=1)]


AttributeValue: TypeAlias = StrictStr | AttributeCandidate


class AttributeClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: Annotated[StrictStr, Field(min_length=1)]
    attributes: list[AttributeValue] = Field(default_factory=list)


class AttributeClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    attribute: AttributeValue
    score: confloat(ge=0.0, le=1.0)
    relevant: bool


class AttributeClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_attributes: list[AttributeValue] = Field(default_factory=list)
    results: list[AttributeClassificationResult] = Field(default_factory=list)
