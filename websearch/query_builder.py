from websearch.models.common_properties import CommonProperties

# try to construct a web query out of given filters/properties
def build_web_query(common_filters: CommonProperties | None) -> str:
    if not common_filters:
        return ""
    parts: list[str] = []
    for value in [
        getattr(common_filters, "color", None),
        getattr(common_filters, "gender", None),
    ]:
        if value:
            parts.append(str(value))
    return " ".join(parts).strip()
