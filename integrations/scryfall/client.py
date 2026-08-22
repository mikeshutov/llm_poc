from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import DEFAULT_TTL, HttpClient, HttpClientError
from integrations.scryfall.models import ScryfallCard, ScryfallCardSearchResult, ScryfallRulingList


class ScryfallClientError(RuntimeError):
    pass


class ScryfallClient:
    def __init__(
        self,
        base_url: str = "https://api.scryfall.com",
        timeout_s: float = 20.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            ttl=ttl,
        )

    def search_cards(
        self,
        query: str,
        *,
        unique: str = "cards",
        order: str = "name",
        dir: str = "auto",
    ) -> ScryfallCardSearchResult:
        normalized_query = (query or "").strip()
        if not normalized_query:
            raise ValueError("Search query must not be empty.")

        params: dict[str, Any] = {
            "q": normalized_query,
            "unique": unique,
            "order": order,
            "dir": dir,
        }
        url = f"{self.base_url}/cards/search"
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise ScryfallClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise ScryfallClientError("Unexpected response from Scryfall API.")
        return ScryfallCardSearchResult.model_validate(payload)

    def get_card_by_name(self, name: str, *, fuzzy: bool = True) -> ScryfallCard:
        normalized_name = (name or "").strip()
        if not normalized_name:
            raise ValueError("Card name must not be empty.")

        params = {"fuzzy": normalized_name} if fuzzy else {"exact": normalized_name}
        url = f"{self.base_url}/cards/named"
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise ScryfallClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise ScryfallClientError("Unexpected card lookup response from Scryfall API.")
        return ScryfallCard.model_validate(payload)

    def get_cards_by_names(self, names: list[str]) -> list[ScryfallCard]:
        normalized_names = list(dict.fromkeys(name.strip() for name in names if name and name.strip()))
        if not normalized_names:
            return []
        if len(normalized_names) > 75:
            raise ValueError("Scryfall collection lookups support at most 75 cards.")

        url = f"{self.base_url}/cards/collection"
        request_payload = {"identifiers": [{"name": name} for name in normalized_names]}
        try:
            payload = self._http.post(url, request_payload)
        except HttpClientError as e:
            raise ScryfallClientError(str(e)) from e
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ScryfallClientError("Unexpected card collection response from Scryfall API.")
        return [ScryfallCard.model_validate(card) for card in payload["data"] if isinstance(card, dict)]

    def get_card_rulings(self, card_id: str) -> ScryfallRulingList:
        normalized_card_id = (card_id or "").strip()
        if not normalized_card_id:
            raise ValueError("Card id must not be empty.")

        url = f"{self.base_url}/cards/{normalized_card_id}/rulings"
        try:
            payload = self._http.get(url)
        except HttpClientError as e:
            raise ScryfallClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise ScryfallClientError("Unexpected rulings response from Scryfall API.")
        return ScryfallRulingList.model_validate(payload)
