from enum import Enum


class ToolName(str, Enum):
    PLAN_AGENT_STEP = "plan_agent_step"
    FINAL_DECISION = "final_decision"
    FIND_PRODUCTS = "find_products"
    GENERATE_RESPONSE = "generate_response"
    GENERIC_WEB_SEARCH = "generic_web_search"
    RESOLVE_CITY_LOCATION = "resolve_city_location"
    GET_HISTORICAL_MONTH_WEATHER = "get_historical_month_weather"
    LIST_PRODUCT_CATEGORIES = "list_product_categories"
    UNKNOWN_INTENT_HANDLER = "unknown_intent_handler"
    WEATHER_ACTION_GUARD = "weather_action_guard"
    REDUNDANT_FETCH_GUARD = "redundant_fetch_guard"
    FALLBACK_STOP_GUARD = "fallback_stop_guard"
