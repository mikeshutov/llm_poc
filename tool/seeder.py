from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from tool.models import RegisteredTool, Tool
from tool.repository.repo_factory import get_tool_registry_repo
from tool.tools import tools


def _to_registered_tool(tool: Tool) -> RegisteredTool:
    return RegisteredTool(
        id=uuid4(),
        name=tool.name,
        version=tool.version,
        description=tool.description,
        result_type=tool.result_type,
        rate_limit_key=tool.rate_limit_key,
        retry_policy=tool.retry_policy,
        rate_limit_policy=tool.rate_limit_policy,
    )


def seed_tools(definitions: Iterable[Tool] = tools) -> list[RegisteredTool]:
    repository = get_tool_registry_repo()
    return repository.sync(_to_registered_tool(tool) for tool in definitions)

