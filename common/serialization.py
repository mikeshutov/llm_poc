from __future__ import annotations

from typing import Any


def prune_empty_prompt_values(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {
            key: prune_empty_prompt_values(item)
            for key, item in value.items()
        }
        return {
            key: item
            for key, item in pruned.items()
            if item not in (None, "", [], {})
        }

    if isinstance(value, list):
        pruned = [prune_empty_prompt_values(item) for item in value]
        return [item for item in pruned if item not in (None, "", [], {})]

    return value
