from enum import Enum

class Intent(str, Enum):
    FIND_PRODUCTS = "find_products"
    GENERAL_INFORMATION = "general_information"
    UNKNOWN = "unknown"

    # future intents we could come up with
    #TRACK_ORDER = "track_order"
    #BRAND_INFO = "brand_info"
    #FAQ = "faq"