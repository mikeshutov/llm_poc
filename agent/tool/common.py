from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def model_to_dict(value: BaseModel | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return value
