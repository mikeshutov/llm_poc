from __future__ import annotations

from capabilities.registry.constants import TOOL_NAME_UPDATE_MEMORY_STATE

COMPACTION_EXCLUDED_TOOLS = frozenset({TOOL_NAME_UPDATE_MEMORY_STATE})


def should_compact_tool(tool_name: str) -> bool:
    """Canonical policy for both persisted records and active history."""
    if tool_name in COMPACTION_EXCLUDED_TOOLS:
        return False
    from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY, UnknownToolError

    try:
        tool = GLOBAL_TOOL_REGISTRY.get(tool_name)
    except UnknownToolError:
        return False
    metadata = getattr(tool, "metadata", None) or {}
    return bool(getattr(tool, "compactable", metadata.get("compactable", True)))


__all__ = [
    "COMPACTION_EXCLUDED_TOOLS",
    "should_compact_tool",
]
