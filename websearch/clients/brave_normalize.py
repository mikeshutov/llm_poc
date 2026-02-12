from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WebSearchItem:
    title: str
    url: Optional[str]
    description: str
    image_url: Optional[str]


_TITLE_KEYS = ("title", "name")
_URL_KEYS = ("url", "link")
_DESC_KEYS = ("description", "snippet", "summary")
_IMAGE_KEYS = ("image", "thumbnail", "thumbnail_url", "thumbnailUrl")


def _first_str(data: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_matching_attribute(data: dict, keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            nested_url = value.get("url")
            if isinstance(nested_url, str) and nested_url.strip():
                return nested_url.strip()
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_web_item(item: dict) -> WebSearchItem:
    return WebSearchItem(
        title=_first_str(item, _TITLE_KEYS),
        url=_first_matching_attribute(item, _URL_KEYS),
        description=_first_str(item, _DESC_KEYS),
        image_url=_first_matching_attribute(item, _IMAGE_KEYS),
    )
