from enum import Enum

class Intent(str, Enum):
    FIND_PRODUCTS = "find_products"
    UNKNOWN = "unknown"

    # future intents we could come up with
    #TRACK_ORDER = "track_order"
    #BRAND_INFO = "brand_info"
    #FAQ = "faq"