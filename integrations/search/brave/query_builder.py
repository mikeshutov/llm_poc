from products.models.product_query import ProductQuery


def build_web_query(product_filters: ProductQuery | None) -> str:
    if not product_filters:
        return ""
    parts: list[str] = []
    for value in [product_filters.color, product_filters.gender]:
        if value:
            parts.append(str(value))
    return " ".join(parts).strip()
