from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL, build_headers
from integrations.wikidata.models import SparqlResult


class WikidataSparqlClientError(RuntimeError):
    pass


class WikidataSparqlClient:
    def __init__(
        self,
        base_url: str = "https://query.wikidata.org/sparql",
        timeout_s: float = 20.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            headers=build_headers(Accept="application/sparql-results+json"),
            ttl=ttl,
        )

    def query(self, sparql: str) -> SparqlResult:
        sparql_norm = (sparql or "").strip()
        if not sparql_norm:
            raise ValueError("SPARQL query must be a non-empty string.")

        try:
            payload = self._http.get(self.base_url, {"query": sparql_norm, "format": "json"})
        except HttpClientError as e:
            raise WikidataSparqlClientError(str(e)) from e

        if not isinstance(payload, dict):
            raise WikidataSparqlClientError("Unexpected non-object response from Wikidata SPARQL endpoint.")

        return SparqlResult.model_validate({
            "sparql": sparql_norm,
            "vars": (payload.get("head") or {}).get("vars", []),
            "bindings": (payload.get("results") or {}).get("bindings", []),
        })

    def select(self, sparql: str) -> list[dict[str, Any]]:
        result = self.query(sparql)
        return result.bindings
