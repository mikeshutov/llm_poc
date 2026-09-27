from __future__ import annotations

import json
from typing import Any

from common.utils import normalize_text
from personalization.profile.models import UserProfile


def stringify_reranker_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return str(value).lower() if isinstance(value, bool) else None
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=True, default=str)
    text = normalize_text(str(value))
    return text or None


def build_reranker_query(goal: str | None, user_profile: UserProfile | None = None) -> str:
    parts: list[str] = []
    normalized_goal = normalize_text(goal) if goal else None
    if normalized_goal:
        parts.append(f"query: {normalized_goal}")
    if user_profile is not None:
        attributes = user_profile.to_prompt_dict().get("user_attributes", {}).get("attributes", [])
        profile_lines = []
        for attribute in attributes:
            attribute_type = stringify_reranker_value(attribute.get("attribute_type"))
            value = stringify_reranker_value(attribute.get("value"))
            if attribute_type and value:
                profile_lines.append(f"{attribute_type}: {value}")
        if profile_lines:
            parts.append("preferences: " + "; ".join(profile_lines))
    return "\n".join(parts)
