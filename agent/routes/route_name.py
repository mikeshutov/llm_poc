from enum import Enum


class RouteName(str, Enum):
    PRODUCTS = "products"
    GENERAL_INFO = "general_info"
    UNSUPPORTED = "unsupported"
