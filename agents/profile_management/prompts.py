from __future__ import annotations


def build_profile_management_rules(attribute_prefixes: str, attribute_suffixes: str) -> str:
    return (
        "Attribute Type Rules:\n"
        f"- Available attribute prefixes: {attribute_prefixes}.\n"
        f"- Available attribute suffixes: {attribute_suffixes}.\n"
        "- Requested or updated attribute types must use the format prefix.suffix such as "
        "food.preference or projects.goal.\n\n"
        "Execution rules:\n"
        "- Do not repeat executed calls.\n"
        "- Do not make one planned tool step depend on another step's output. Each step must "
        "stand on its own inputs for now.\n"
        "- Keep plans to one sentence."
    )


__all__ = ["build_profile_management_rules"]
