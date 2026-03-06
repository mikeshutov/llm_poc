from dataclasses import dataclass


@dataclass(frozen=True)
class GeocodedLocation:
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str | None
