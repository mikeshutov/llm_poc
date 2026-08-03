from __future__ import annotations

from typing import Literal, TypeAlias

ATTRIBUTE_CATEGORIES = (
    "career",
    "food",
    "technology",
    "projects",
    "media",
    "fitness",
)

ATTRIBUTE_QUALIFIERS = (
    "likes",
    "dislikes",
    "favorites",
    "interests",
    "skills",
)

ATTRIBUTE_TYPE_VALUES = tuple(
    f"{category}.{qualifier}"
    for category in ATTRIBUTE_CATEGORIES
    for qualifier in ATTRIBUTE_QUALIFIERS
)

UserAttributeType: TypeAlias = Literal[*ATTRIBUTE_TYPE_VALUES]
ATTRIBUTE_TYPE_DESCRIPTION = ", ".join(ATTRIBUTE_TYPE_VALUES)
