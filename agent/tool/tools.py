from agent.tool.brave_news_search import news_search
from agent.tool.exchange_rates_lookup import exchange_rates_lookup
from agent.tool.exchange_rates_time_series import exchange_rates_time_series
from agent.tool.find_products import find_products
from agent.tool.generic_web_search import generic_web_search
from agent.tool.get_historical_month_weather import get_historical_month_weather
from agent.tool.list_product_categories import list_product_categories
from agent.tool.public_holidays_lookup import public_holidays_lookup
from agent.tool.resolve_city_location import resolve_city_location
from agent.tool.structured_facts_lookup import structured_facts_lookup
from agent.tool.wikipedia_search import wikipedia_search

tools = [
    find_products,
    resolve_city_location,
    get_historical_month_weather,
    public_holidays_lookup,
    exchange_rates_lookup,
    exchange_rates_time_series,
    generic_web_search,
    news_search,
    wikipedia_search,
    structured_facts_lookup,
    list_product_categories,
]
