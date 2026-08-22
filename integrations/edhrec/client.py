from __future__ import annotations

from datetime import timedelta
import re
import unicodedata

from common.http import DEFAULT_TTL, HttpClient, HttpClientError
from integrations.edhrec.models import EdhrecCommanderPage

EDHREC_API_BASE_URL = "https://json.edhrec.com"
EDHREC_WEBSITE_BASE_URL = "https://edhrec.com"
EDHREC_COMMANDER_URL_TEMPLATE = "https://edhrec.com/commanders/{slug}"
EDHREC_CARD_URL_TEMPLATE = "https://edhrec.com{path}"


class EdhrecClientError(RuntimeError):
    pass


def slugify_commander_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise ValueError("Commander name must not be empty.")
    return normalized


class EdhrecClient:
    def __init__(
        self,
        base_url: str = EDHREC_API_BASE_URL,
        timeout_s: float = 20.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(timeout_s=timeout_s, ttl=ttl)

    def get_commander_page(self, commander_name: str) -> tuple[str, EdhrecCommanderPage]:
        slug = slugify_commander_name(commander_name)
        url = f"{self.base_url}/pages/commanders/{slug}.json"
        try:
            payload = self._http.get(url)
        except HttpClientError as e:
            raise EdhrecClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise EdhrecClientError("Unexpected response from EDHREC endpoint.")
        return slug, EdhrecCommanderPage.model_validate(payload)
