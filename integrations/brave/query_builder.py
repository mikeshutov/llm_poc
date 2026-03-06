from integrations.models.common_properties import CommonProperties


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
