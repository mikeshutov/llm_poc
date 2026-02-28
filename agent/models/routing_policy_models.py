from dataclasses import dataclass, field

from agent.routes.route_name import RouteName
from agent.tools.tool_name import ToolName
from intent_layer.models.intent import Intent
from intent_layer.models.parsed_request import ParsedRequest

SENSITIVE_SAFETY_FLAGS = {
    "self_harm",
    "suicide",
    "kill_myself",
    "bomb",
    "weapon",
    "malware",
    "ransomware",
    "exploit",
    "bypass",
    "credential_stuffing",
    "ddos",
}


def _fallback_sensitive_terms() -> set[str]:
    terms: set[str] = set()
    for flag in SENSITIVE_SAFETY_FLAGS:
        terms.add(flag.replace("_", " "))
        terms.add(flag.replace("_", "-"))
    return terms


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    supported: bool
    reason: str
    allowed_tools: set[ToolName]


@dataclass(frozen=True)
class UnknownHandlingPolicy:
    route_tool_allowlist: dict[RouteName, set[ToolName]] = field(
        default_factory=lambda: {
            RouteName.PRODUCTS: {
                ToolName.LIST_PRODUCT_CATEGORIES,
                ToolName.FIND_PRODUCTS,
                ToolName.GENERATE_RESPONSE,
                ToolName.RESOLVE_CITY_LOCATION,
                ToolName.GET_HISTORICAL_MONTH_WEATHER,
            },
            RouteName.GENERAL_INFO: {
                ToolName.GENERIC_WEB_SEARCH,
                ToolName.GENERATE_RESPONSE,
            },
            RouteName.UNSUPPORTED: set(),
        }
    )

    def allow(self, route: RouteName) -> bool:
        return route in self.route_tool_allowlist and route != RouteName.UNSUPPORTED

    def _is_sensitive_unsupported(self, parsed_query: ParsedRequest, query_text: str) -> bool:
        safety_flags = parsed_query.safety_flags or []
        normalized_flags = {
            str(flag).strip().lower().replace("-", "_").replace(" ", "_")
            for flag in safety_flags
            if str(flag).strip()
        }
        if normalized_flags & SENSITIVE_SAFETY_FLAGS:
            return True

        lowered = query_text.lower()
        return any(term in lowered for term in _fallback_sensitive_terms())

    def decide(self, parsed_query: ParsedRequest) -> RouteDecision:
        query_text = (parsed_query.query_details.query_text if parsed_query.query_details else "") or ""
        if parsed_query.intent == Intent.FIND_PRODUCTS:
            route = RouteName.PRODUCTS
            reason = "Intent mapped to product-search workflow."
        elif parsed_query.intent == Intent.GENERAL_INFORMATION:
            route = RouteName.GENERAL_INFO
            reason = "Intent mapped to general-information workflow."
        elif (
            parsed_query.intent == Intent.UNKNOWN
            and query_text.strip()
            and not self._is_sensitive_unsupported(parsed_query, query_text)
        ):
            route = RouteName.GENERAL_INFO
            reason = "Unknown intent with usable query text; defaulting to general-information route."
        else:
            route = RouteName.UNSUPPORTED
            reason = "Intent is unknown or unsupported."

        supported = self.allow(route)
        if not supported and route != RouteName.UNSUPPORTED:
            route = RouteName.UNSUPPORTED
            reason = "Request blocked by routing policy."

        return RouteDecision(
            route=route,
            supported=supported,
            reason=reason,
            allowed_tools=set(self.route_tool_allowlist.get(route, set())),
        )
