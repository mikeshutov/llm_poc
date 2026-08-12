from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel


def is_meaningful_prompt_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip()
        return normalized not in {"", "{}", "[]", '""'}
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def prune_empty_prompt_values(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {
            key: prune_empty_prompt_values(item)
            for key, item in value.items()
        }
        return {
            key: item
            for key, item in pruned.items()
            if is_meaningful_prompt_value(item)
        }

    if isinstance(value, list):
        pruned = [prune_empty_prompt_values(item) for item in value]
        return [item for item in pruned if is_meaningful_prompt_value(item)]

    return value


def sanitize_for_json_storage(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return sanitize_for_json_storage(value.model_dump())
    if is_dataclass(value):
        return sanitize_for_json_storage(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): sanitize_for_json_storage(item)
            for key, item in value.items()
            if not str(key).endswith("_embedding")
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json_storage(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
