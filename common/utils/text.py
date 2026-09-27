from __future__ import annotations


def normalize_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def normalize_values(values: object) -> list[str]:
    """Normalize an API value that may be a scalar or a list of values."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [cleaned for value in values if (cleaned := normalize_text(str(value)))]


def unique_normalized_values(
    values: object,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Normalize values, remove case-insensitive duplicates, and optionally exclude prefixes."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in normalize_values(values):
        if any(value.casefold().startswith(prefix.casefold()) for prefix in exclude_prefixes):
            continue
        normalized = value.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique
