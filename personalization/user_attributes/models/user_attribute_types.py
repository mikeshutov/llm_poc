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
    "goals",
)

ATTRIBUTE_TYPE_VALUES = tuple(
    f"{category}.{qualifier}"
    for category in ATTRIBUTE_CATEGORIES
    for qualifier in ATTRIBUTE_QUALIFIERS
)

UserAttributeType: TypeAlias = Literal[*ATTRIBUTE_TYPE_VALUES]
ATTRIBUTE_TYPE_DESCRIPTION = ", ".join(ATTRIBUTE_TYPE_VALUES)
ATTRIBUTE_TYPE_FORMAT_DESCRIPTION = (
    "Use the format `prefix.suffix` where prefix is one of "
    f"{', '.join(ATTRIBUTE_CATEGORIES)} and suffix is one of {', '.join(ATTRIBUTE_QUALIFIERS)}."
)
ATTRIBUTE_TYPE_EXAMPLE_DESCRIPTION = (
    "Examples: `food.likes`, `projects.goals`, `technology.skills`."
)
ATTRIBUTE_TYPE_COMPACT_DESCRIPTION = (
    f"{ATTRIBUTE_TYPE_FORMAT_DESCRIPTION} {ATTRIBUTE_TYPE_EXAMPLE_DESCRIPTION}"
)
