from __future__ import annotations


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
