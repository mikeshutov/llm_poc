from __future__ import annotations

from langchain_core.tools import tool
from requests.exceptions import RequestException

from integrations.ip_api import IpApiClient, IpLocation

_ip_api_client = IpApiClient()


@tool(
    "get_caller_location",
    description="""
Get the approximate geographic location of the caller based on their IP address.

Returns country, region, city, timezone, latitude, and longitude.

Example valid call:
{}
""",
)
def get_caller_location() -> IpLocation | str:
    try:
        return _ip_api_client.get_location()
    except RequestException as e:
        return f"Location service unavailable: {e}"
