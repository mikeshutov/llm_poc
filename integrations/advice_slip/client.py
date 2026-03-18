from __future__ import annotations

from datetime import timedelta

from common.http import HttpClient, HttpClientError
from integrations.advice_slip.models import AdviceSlip


class AdviceSlipClientError(RuntimeError):
    pass


class AdviceSlipClient:
    def __init__(
        self,
        base_url: str = "https://api.adviceslip.com",
        timeout_s: float = 10.0,
        ttl: timedelta = timedelta(seconds=30),
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            
            ttl=ttl,
        )

    def _extract_slip(self, payload: dict) -> AdviceSlip:
        slip = payload.get("slip")
        if not isinstance(slip, dict):
            raise AdviceSlipClientError("Unexpected response from Advice Slip API.")
        return AdviceSlip.model_validate(slip)

    def random(self) -> AdviceSlip:
        url = f"{self.base_url}/advice"
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise AdviceSlipClientError(str(e)) from e
        return self._extract_slip(payload)

    def search(self, query: str) -> list[AdviceSlip]:
        q = (query or "").strip()
        if not q:
            raise ValueError("Search query must not be empty.")
        url = f"{self.base_url}/advice/search/{q}"
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise AdviceSlipClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise AdviceSlipClientError("Unexpected response from Advice Slip API.")
        slips = payload.get("slips")
        if not isinstance(slips, list):
            return []
        return [AdviceSlip.model_validate(s) for s in slips if isinstance(s, dict)]
