from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.repository.product_repository import ProductRepository
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_LIST_PRODUCT_CATEGORIES
from tool.constants import TOOL_RESULT_TYPE_PRODUCT_CATEGORIES


class ListProductCategoriesArgs(BaseModel):
    limit: int = Field(
        default=200,
        description="Maximum number of categories to return.",
        ge=1,
    )


def _tool_result(result: list[str]) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for category in result:
        title = category.strip()
        hydrated = HydratedEvidence(
            item_id=title,
            tool_name=TOOL_NAME_LIST_PRODUCT_CATEGORIES,
            title=title,
            summary=f"Available product category: {title}.",
            source=TOOL_NAME_LIST_PRODUCT_CATEGORIES,
            entity_type=TOOL_RESULT_TYPE_PRODUCT_CATEGORIES,
            metadata={},
            raw_payload=category,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata={},
            )
        )
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)


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
