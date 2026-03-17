from __future__ import annotations

from datetime import timedelta

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.meal_db.models import Meal, MealSearchResult


class MealDbClientError(RuntimeError):
    pass


class MealDbClient:
    def __init__(
        self,
        base_url: str = "https://www.themealdb.com/api/json/v1/1",
        timeout_s: float = 20.0,
        user_agent: str = "POCProductSearch/1.0 (MealDB client)",
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
            raise MealDbClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise MealDbClientError("Unexpected response from MealDB API.")
        return payload

    def search(self, query: str) -> MealSearchResult:
        q = (query or "").strip()
        if not q:
            raise ValueError("Search query must not be empty.")
        payload = self._get("search.php", {"s": q})
        return MealSearchResult(meals=[
            Meal.model_validate(item)
            for item in (payload.get("meals") or [])
            if isinstance(item, dict)
        ])

    def lookup(self, meal_id: str) -> Meal | None:
        payload = self._get("lookup.php", {"i": meal_id})
        meals = payload.get("meals") or []
        return Meal.model_validate(meals[0]) if meals and isinstance(meals[0], dict) else None

    def random(self) -> Meal | None:
        payload = self._get("random.php", {})
        meals = payload.get("meals") or []
        return Meal.model_validate(meals[0]) if meals and isinstance(meals[0], dict) else None
