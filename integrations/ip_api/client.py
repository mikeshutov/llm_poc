from __future__ import annotations

from datetime import timedelta

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.ip_api.models import IpLocation


class IpApiClientError(RuntimeError):
    pass


class IpApiClient:
    def __init__(
        self,
        base_url: str = "http://ip-api.com/json",
        timeout_s: float = 10.0,
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(timeout_s=timeout_s, ttl=ttl)

    def get_location(self, ip: str | None = None) -> IpLocation:
        url = f"{self.base_url}/{ip}" if ip else self.base_url
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise IpApiClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise IpApiClientError("Unexpected response from ip-api.")
        location = IpLocation.model_validate(payload)
        if location.status != "success":
            raise IpApiClientError(f"ip-api returned status: {location.status}")
        return location
