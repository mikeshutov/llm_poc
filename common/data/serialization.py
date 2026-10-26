from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel


def is_meaningful_prompt_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
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


def bounded_json_value(
    value: Any,
    *,
    max_bytes: int = 64_000,
    max_depth: int = 8,
    max_items: int = 100,
    max_string: int = 4_000,
) -> Any:
    """Return JSON-safe data bounded before it reaches JSONB or tracing."""
    max_bytes = max(256, int(max_bytes))

    def project(item: Any, depth: int) -> Any:
        if depth >= max_depth:
            return "[truncated: maximum depth]"
        if isinstance(item, BaseModel):
            return project(item.model_dump(), depth + 1)
        if is_dataclass(item):
            return project(asdict(item), depth + 1)
        if isinstance(item, UUID):
            return str(item)
        if isinstance(item, str):
            return item if len(item) <= max_string else item[:max_string] + "…"
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for index, (key, child) in enumerate(item.items()):
                if str(key).endswith("_embedding"):
                    continue
                if index >= max_items:
                    result["_truncated_items"] = True
                    break
                result[str(key)] = project(child, depth + 1)
            return result
        if isinstance(item, (list, tuple, set)):
            result = [project(child, depth + 1) for child in list(item)[:max_items]]
            if len(item) > max_items:
                result.append("[truncated: additional items]")
            return result
        return item

    projected = project(value, 0)
    rendered = json.dumps(projected, ensure_ascii=True, separators=(",", ":"), default=str)
    if len(rendered.encode("utf-8")) <= max_bytes:
        return projected
    preview = rendered.encode("utf-8")[: max_bytes - 80].decode("utf-8", errors="ignore")
    return {"_truncated": True, "_preview": preview}
