from uuid import uuid4

from capabilities.registry.models import RegisteredTool, RateLimitPolicy
from capabilities.registry.discovery_metadata import DISCOVERY_METADATA
from capabilities.registry.seeder import _to_registered_tool
from capabilities.registry.tools import tools
from capabilities.registry.models.tool_definition import Tool
from capabilities.retry.policy import RetryPolicy
import capabilities.registry.repository.tool_registry_repository as registry_module


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
    runtime_tool = Tool(lambda: None, result_type="calculation", version=3)
    runtime_tool.fn.name = "math_calculate"
    runtime_tool.fn.description = "A test tool"

    definition = _to_registered_tool(runtime_tool)

    assert definition.name == "math_calculate"
    assert definition.version == 3
    assert definition.description.startswith("A test tool")
    assert definition.result_type == "calculation"
    assert "calculate" in definition.embedding_text
    assert "mathematical" in definition.embedding_text


def test_registered_tool_defaults_to_active_version_one() -> None:
    definition = RegisteredTool(id=uuid4(), name="test_tool")

    assert definition.version == 1
    assert definition.is_active is True


def test_registry_upsert_embeds_explicit_name_plus_discovery_text(monkeypatch) -> None:
    definition = RegisteredTool(
        id=uuid4(),
        name="search_files",
        version=2,
        description="Detailed planner description with examples.",
        embedding_text="search_files\nUploaded file discovery\nSearch uploaded files by semantic relevance.",
    )
    captured: list[str] = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query, params):
            captured.append(params[3])

        def fetchone(self):
            return {
                "id": definition.id,
                "name": definition.name,
                "version": definition.version,
                "description": definition.description,
                "result_type": definition.result_type,
                "rate_limit_key": None,
                "retry_policy": {},
                "rate_limit_policy": None,
                "is_active": True,
                "metadata": {},
                "created_at": None,
                "updated_at": None,
            }

    class Connection:
        def cursor(self, **kwargs):
            return Cursor()

    monkeypatch.setattr(registry_module, "embed_text", lambda value: captured.append(value) or [0.1])
    repository = object.__new__(registry_module.ToolRegistryRepository)
    repository._conn = Connection()

    repository.upsert(definition)

    assert captured[0] == definition.embedding_text
    assert "Detailed planner description" not in captured[0]


def test_all_registered_tools_have_versioned_discovery_metadata() -> None:
    tool_names = {tool.name for tool in tools}

    assert tool_names == set(DISCOVERY_METADATA)
    assert all(tool.version == tool.discovery_metadata.version for tool in tools)
    assert all(metadata.version >= 1 for metadata in DISCOVERY_METADATA.values())


def test_embedding_text_is_concise_and_distinguishes_products_from_files() -> None:
    definitions = {}
    for tool in tools:
        definition = _to_registered_tool(tool)
        definitions[definition.name] = definition

    product_text = definitions["products_search"].embedding_text
    web_product_text = definitions["products_search"].embedding_text
    file_text = definitions["file_search"].embedding_text
    known_file_text = definitions["file_search_details"].embedding_text

    assert "products_search" in product_text
    assert "internal catalog" in product_text
    assert "products_search" in web_product_text
    assert "web shopping" in web_product_text
    assert "file_search" in file_text
    assert "uploaded files" in file_text
    assert "file_search_details" in known_file_text
    assert "known uploaded file" in known_file_text
    assert "Example valid call" not in product_text
    assert "Required fields" not in product_text
