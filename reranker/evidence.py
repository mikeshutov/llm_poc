from __future__ import annotations

import json
import os
from typing import Any

from common.utils import normalize_text
from personalization.profile.models import UserProfile
from reranker.models.candidate import Candidate

RERANKER_CANDIDATE_TOKEN_BUDGET = max(64, int(os.getenv("RERANKER_CANDIDATE_TOKEN_BUDGET", "384")))


def _stringify(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return str(value).lower() if isinstance(value, bool) else None
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=True, default=str)
    text = normalize_text(str(value))
    return text or None


def _field_lines(candidate: Candidate) -> list[str]:
    lines: list[str] = []
    if candidate.title:
        lines.append(f"title: {candidate.title}")

    content = candidate.content or {}
    excluded_content = {"url", "image_url", "source", "retrieval_distance"}
    preferred_content = ("name", "summary", "description", "text")
    keys = [key for key in preferred_content if key in content]
    content_keys = set(content.model_dump().keys()) if hasattr(content, "model_dump") else set(content)
    keys.extend(sorted(key for key in content_keys if key not in keys and key not in excluded_content))
    for key in keys:
        if key == "name" and candidate.title and _stringify(content.get(key)) == _stringify(candidate.title):
            continue
        value = _stringify(content.get(key))
        if value:
            lines.append(f"{key}: {value}")

    excluded_attributes = {
        "source",
        "retrieval_distance",
        "retrieval_score",
        "embedding",
        "flags",
        "reasons",
    }
    for key in sorted(candidate.attributes or {}):
        if key in excluded_attributes:
            continue
        value = _stringify(candidate.attributes[key])
        if value:
            lines.append(f"{key}: {value}")
    return lines


def build_candidate_evidence(candidate: Candidate, *, token_budget: int | None = None) -> str:
    budget = max(64, token_budget or RERANKER_CANDIDATE_TOKEN_BUDGET)
    max_chars = budget * 4
    lines = _field_lines(candidate)
    result: list[str] = []
    remaining = max_chars
    for line in lines:
        separator = 1 if result else 0
        available = remaining - separator
        if available <= 0:
            break
        if len(line) > available:
            result.append(line[:available].rstrip())
            break
        result.append(line)
        remaining -= len(line) + separator
    return "\n".join(result)


def build_reranker_query(goal: str | None, user_profile: UserProfile | None = None) -> str:
    parts: list[str] = []
    normalized_goal = normalize_text(goal) if goal else None
    if normalized_goal:
        parts.append(f"query: {normalized_goal}")
    if user_profile is not None:
        attributes = user_profile.to_prompt_dict().get("user_attributes", {}).get("attributes", [])
        profile_lines = []
        for attribute in attributes:
            attribute_type = _stringify(attribute.get("attribute_type"))
            value = _stringify(attribute.get("value"))
            if attribute_type and value:
                profile_lines.append(f"{attribute_type}: {value}")
        if profile_lines:
            parts.append("preferences: " + "; ".join(profile_lines))
    return "\n".join(parts)
