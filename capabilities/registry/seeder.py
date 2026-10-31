from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from capabilities.registry.models import RegisteredTool, Tool
from capabilities.registry.discovery_metadata import validate_tool_discovery_metadata
from capabilities.registry.repository.repo_factory import get_tool_registry_repo
from capabilities.registry.tools import tools


def _to_registered_tool(tool: Tool) -> RegisteredTool:
    discovery = tool.discovery_metadata
    return RegisteredTool(
        id=uuid4(),
        name=tool.name,
        version=tool.version,
        description=tool.description,
        result_type=tool.result_type,
        rate_limit_key=tool.rate_limit_key,
        retry_policy=tool.retry_policy,
        rate_limit_policy=tool.rate_limit_policy,
        metadata={
            "title": discovery.title,
            "category": discovery.category,
            "scope_terms": list(discovery.scope_terms),
            "result_fields": list(tool.available_result_fields),
        },
        embedding_text=tool.embedding_text,
    )


def seed_tools(definitions: Iterable[Tool] = tools) -> list[RegisteredTool]:
    definitions = tuple(definitions)
    validate_tool_discovery_metadata({tool.name for tool in definitions})
    version_mismatches = [
        f"{tool.name}: tool={tool.version}, discovery={tool.discovery_metadata.version}"
        for tool in definitions
        if tool.version != tool.discovery_metadata.version
    ]
    if version_mismatches:
        raise ValueError("Tool/discovery version mismatch: " + ", ".join(version_mismatches))
    repository = get_tool_registry_repo()
    return repository.sync(_to_registered_tool(tool) for tool in definitions)

