from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.repository.product_repository import ProductRepository
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_LIST_PRODUCT_CATEGORIES
from tool.constants import TOOL_RESULT_TYPE_PRODUCT_CATEGORIES


class ListProductCategoriesArgs(BaseModel):
    limit: int = Field(
        default=200,
        description="Maximum number of categories to return.",
        ge=1,
    )


def _tool_result(result: list[str]) -> ToolResult:
    evidence: list[EvidenceView] = []
    for category in result:
        title = category.strip()
        hydrated = EvidenceView(
            item_id=title,
            tool_name=TOOL_NAME_LIST_PRODUCT_CATEGORIES,
            title=title,
            summary=f"Available product category: {title}.",
            source=TOOL_NAME_LIST_PRODUCT_CATEGORIES,
            entity_type=TOOL_RESULT_TYPE_PRODUCT_CATEGORIES,
            llm_metadata={},
            raw_payload=category,
        )
        evidence.append(hydrated)
    return ToolResult(result=result, evidence=evidence)


@tool(
    TOOL_NAME_LIST_PRODUCT_CATEGORIES,
    args_schema=ListProductCategoriesArgs,
    description="""
Return available product categories from the internal catalog.

Optional fields:
- limit (integer)

Example valid call:
{
  "limit": 200
}
""",
)
def list_product_categories(limit: int = 200) -> ToolResult:
    return _tool_result(ProductRepository().list_categories(limit=limit))
