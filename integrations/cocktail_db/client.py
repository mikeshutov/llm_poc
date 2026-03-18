from __future__ import annotations

from datetime import timedelta

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.cocktail_db.models import Cocktail, CocktailSearchResult


class CocktailDbClientError(RuntimeError):
    pass


class CocktailDbClient:
    def __init__(
        self,
        base_url: str = "https://www.thecocktaildb.com/api/json/v1/1",
        timeout_s: float = 20.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            
            ttl=ttl,
        )

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise CocktailDbClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise CocktailDbClientError("Unexpected response from CocktailDB API.")
        return payload

    def search(self, query: str) -> CocktailSearchResult:
        q = (query or "").strip()
        if not q:
            raise ValueError("Search query must not be empty.")
        payload = self._get("search.php", {"s": q})
        return CocktailSearchResult(drinks=[
            Cocktail.model_validate(item)
            for item in (payload.get("drinks") or [])
            if isinstance(item, dict)
        ])

    def random(self) -> Cocktail | None:
        payload = self._get("random.php", {})
        drinks = payload.get("drinks") or []
        return Cocktail.model_validate(drinks[0]) if drinks and isinstance(drinks[0], dict) else None
