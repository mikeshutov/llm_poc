from llm.models.tool_call import ToolCall

from agent.tools.tool_name import ToolName

DISCOVERY_TOOL_NAMES = {
    ToolName.LIST_PRODUCT_CATEGORIES.value,
    ToolName.RESOLVE_CITY_LOCATION.value,
    ToolName.GET_HISTORICAL_MONTH_WEATHER.value,
}


def discovery_requested(tool_calls: list[ToolCall]) -> bool:
    return any(tc.name in DISCOVERY_TOOL_NAMES for tc in tool_calls)


def ordered_tool_calls(tool_calls: list[ToolCall]) -> list[ToolCall]:
    discovery_calls: list[ToolCall] = []
    retrieval_calls: list[ToolCall] = []
    final_decision_calls: list[ToolCall] = []
    other_calls: list[ToolCall] = []
    for tc in tool_calls:
        if tc.name == ToolName.FINAL_DECISION.value:
            final_decision_calls.append(tc)
        elif tc.name in DISCOVERY_TOOL_NAMES:
            discovery_calls.append(tc)
        elif tc.name == ToolName.FIND_PRODUCTS.value:
            retrieval_calls.append(tc)
        else:
            other_calls.append(tc)
    return [*discovery_calls, *other_calls, *retrieval_calls, *final_decision_calls]

