from html.parser import HTMLParser
from typing import Any


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li"}:
            self.parts.append(" ")


def html_to_plain_text(value: str) -> str:
    """Strip provider markup and normalize whitespace for LLM-facing text."""
    extractor = _HtmlTextExtractor()
    extractor.feed(value)
    extractor.close()
    return " ".join("".join(extractor.parts).split())


def normalize_html_values(value: Any) -> Any:
    """Normalize strings nested in LLM-facing structured data."""
    if isinstance(value, str):
        return html_to_plain_text(value)
    if isinstance(value, list):
        return [normalize_html_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_html_values(item) for item in value)
    if isinstance(value, dict):
        return {key: normalize_html_values(item) for key, item in value.items()}
    return value
