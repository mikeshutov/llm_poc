from __future__ import annotations

from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_CATEGORIES, ATTRIBUTE_KINDS

def build_profile_management_rules() -> str:
    attribute_prefixes = ", ".join(ATTRIBUTE_CATEGORIES)
    attribute_suffixes = ", ".join(ATTRIBUTE_KINDS)
    return (
        "Attribute Type Rules:\n"
        f"- Available attribute prefixes: {attribute_prefixes}.\n"
        f"- Available attribute suffixes: {attribute_suffixes}.\n"
        "- Requested or updated attribute types must use the format prefix.suffix such as "
        "food.preference or projects.goal.\n\n"
        "Execution rules:\n"
        "- Map a requested tone to the closest available supported tone/formality value; "
        "do not invent new tone values. Only update the durable profile preference when "
        "the user clearly asks for that preference to be remembered.\n"
        "- For any clear durable tone preference, execute the corresponding profile mutation "
        "before returning; do not report success without a completed update or an explicit "
        "unchanged result.\n"
    )


__all__ = [
    "build_profile_management_rules",
]
