from agent.runtime.governors.base import LoopGovernor
from agent.runtime.governors.general_info_governor import GeneralInfoGovernor
from agent.runtime.governors.products_governor import ProductsGovernor
from agent.runtime.route_kind import RouteKind


def governor_for(route_kind: RouteKind) -> LoopGovernor:
    if route_kind == RouteKind.PRODUCTS:
        return ProductsGovernor()
    return GeneralInfoGovernor()
