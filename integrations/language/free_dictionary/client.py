from __future__ import annotations

from datetime import timedelta

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.language.free_dictionary.models import DictionaryEntry


class FreeDictionaryClientError(RuntimeError):
    pass


class FreeDictionaryClient:
    def __init__(
        self,
        base_url: str = "https://api.dictionaryapi.dev/api/v2/entries",
        timeout_s: float = 10.0,
        user_agent: str = "POCProductSearch/1.0 (FreeDictionary client)",
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def define(self, word: str, language: str = "en") -> list[DictionaryEntry]:
        w = (word or "").strip()
        if not w:
            raise ValueError("Word must not be empty.")

        url = f"{self.base_url}/{language}/{w}"
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise FreeDictionaryClientError(str(e)) from e
        if not isinstance(payload, list):
            raise FreeDictionaryClientError("Unexpected response from Free Dictionary API.")
        return [DictionaryEntry.model_validate(item) for item in payload if isinstance(item, dict)]
