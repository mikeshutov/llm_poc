from __future__ import annotations

from collections.abc import Sequence


def format_prompt_bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) + "\n"


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            s = s.lstrip()
            if s.startswith("json"):
                s = s[4:].lstrip()
    return s.strip()


def normalize_string_list(values: Sequence[object], *, field_name: str = "value") -> list[str]:
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one non-empty string.")
    return normalized
