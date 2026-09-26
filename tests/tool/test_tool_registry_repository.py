from uuid import uuid4

from tool.models import RegisteredTool, RateLimitPolicy, RetryPolicy
from tool.seeder import _to_registered_tool
from tool.models.tool_definition import Tool


def test_registered_tool_validates_version_and_policies() -> None:
    definition = RegisteredTool(
        id=uuid4(),
        name="search_products",
        version=2,
        retry_policy=RetryPolicy(max_attempts=3),
        rate_limit_policy=RateLimitPolicy(max_requests=1, window_seconds=1.0),
    )

    assert definition.version == 2
    assert definition.retry_policy.max_attempts == 3
    assert definition.rate_limit_policy is not None


def test_seed_definition_copies_runtime_tool_metadata() -> None:
    runtime_tool = Tool(lambda: None, result_type="products", version=3)
    runtime_tool.fn.name = "test_tool"
    runtime_tool.fn.description = "A test tool"

    definition = _to_registered_tool(runtime_tool)

    assert definition.name == "test_tool"
    assert definition.version == 3
    assert definition.description == "A test tool"
    assert definition.result_type == "products"


def test_registered_tool_defaults_to_active_version_one() -> None:
    definition = RegisteredTool(id=uuid4(), name="test_tool")

    assert definition.version == 1
    assert definition.is_active is True
