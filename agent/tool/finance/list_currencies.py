from __future__ import annotations

from langchain_core.tools import tool
from requests.exceptions import RequestException

from integrations.coinbase import CoinbaseClient, Currency

_coinbase_client = CoinbaseClient()


@tool(
    "list_currencies",
    description="""
List all supported fiat and crypto currencies from Coinbase, including their names and IDs.

Useful for resolving currency codes or discovering what currencies are available.

Example valid call:
{}
""",
)
def list_currencies() -> list[Currency] | str:
    try:
        return _coinbase_client.get_currencies()
    except RequestException as e:
        return f"Coinbase API unavailable: {e}"
