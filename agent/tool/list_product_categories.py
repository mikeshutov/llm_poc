from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.repository.product_repository import ProductRepository


class ListProductCategoriesArgs(BaseModel):
    limit: int = Field(
        default=200,
        description="Maximum number of categories to return.",
        ge=1,
    )


@tool(
    "list_product_categories",
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
def list_product_categories(limit: int = 200) -> list[str]:
    return ProductRepository().list_categories(limit=limit)
