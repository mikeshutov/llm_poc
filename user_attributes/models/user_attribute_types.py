from __future__ import annotations

from typing import Literal, get_args

UserAttributeType = Literal[
    "like",
    "dislike",
    "preference",
    "fact",
    "constraint",
    "instruction",
    "relationship",
    "trait",
    "other",
]

ATTRIBUTE_TYPE_VALUES = tuple(get_args(UserAttributeType))
ATTRIBUTE_TYPE_DESCRIPTION = ", ".join(ATTRIBUTE_TYPE_VALUES)
