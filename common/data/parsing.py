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


def repair_common_json_issues(s: str) -> str:
    if not s:
        return s

    repaired: list[str] = []
    in_string = False
    escaped = False

    for index, char in enumerate(s):
        if in_string:
            repaired.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            repaired.append(char)
            continue

        if char == ";":
            previous_significant = next((c for c in reversed(repaired) if not c.isspace()), "")
            next_significant = next((c for c in s[index + 1:] if not c.isspace()), "")
            if previous_significant and next_significant == '"':
                repaired.append(",")
                continue

        if char in "}]":
            trailing_whitespace: list[str] = []
            while repaired and repaired[-1].isspace():
                trailing_whitespace.append(repaired.pop())
            if repaired and repaired[-1] == ",":
                repaired.pop()
            repaired.extend(reversed(trailing_whitespace))

        repaired.append(char)

    return "".join(repaired)


def normalize_string_list(values: Sequence[object], field_name: str = "value") -> list[str]:
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one non-empty string.")
    return normalized
