from dataclasses import dataclass
from typing import Any

from agent.models.tool_call_trace import ToolCallTrace
from agent.models.tool_call_status import ToolCallStatus
from agent.tools.tool_name import ToolName
from intent_layer.models.parsed_request import ParsedRequest
from products.models.product_search_results import ProductSearchResults
from websearch.models.common_properties import CommonProperties

MIN_INTERNAL_ONLY_TURNS_BEFORE_WEB_FALLBACK = 5


@dataclass(frozen=True)
class _LastRetrievalOutput:
    internal_count: int
    external_count: int


@dataclass
class _AgentLoopState:
    turn_index: int
    max_turns: int
    goal: str
    goal_reached: bool
    current_query_text: str
    original_query_text: str
    refined_query_text: str
    query_refinement_reason: str | None
    turn_stop_reason: str | None
    current_common_filters: CommonProperties | None
    call_index: int

    def begin_turn(self, turn_index: int) -> None:
        self.turn_index = turn_index
        self.turn_stop_reason = None

    def mark_stop(self, stop_reason: str, goal_reached: bool) -> None:
        self.turn_stop_reason = stop_reason
        self.goal_reached = goal_reached


@dataclass
class _AgentState:
    loop_state: _AgentLoopState
    tool_traces: list[ToolCallTrace]
    iterations_used: int
    last_product_response: ProductSearchResults
    last_retrieval_output: _LastRetrievalOutput
    last_weather_context: dict | None
    last_retrieval_signature: str | None
    retrieval_called_this_turn: bool
    retrieval_from_planner: bool
    last_planner_tool_calls: list[dict[str, Any]]
    last_discovery_tools: list[str]
    discovery_happened_this_iteration: bool
    weather_fetched_this_turn: bool
    weather_fetched_last_turn: bool
    response_status: str | None
    response_error: str | None
    tool_history: list[dict[str, Any]]
    iteration_history: list[dict[str, Any]]
    last_generic_search_payload: dict[str, Any] | None
    last_resolved_location: dict[str, Any] | None

    @classmethod
    def new(cls, parsed_query: ParsedRequest, max_turns: int) -> "_AgentState":
        query_text = parsed_query.query_details.query_text if parsed_query.query_details else ""
        return cls(
            loop_state=_AgentLoopState(
                turn_index=0,
                max_turns=max_turns,
                goal="Find products that match the user's request",
                goal_reached=False,
                current_query_text=query_text,
                original_query_text=query_text,
                refined_query_text=query_text,
                query_refinement_reason=None,
                turn_stop_reason=None,
                current_common_filters=parsed_query.common_properties,
                call_index=0,
            ),
            tool_traces=[],
            iterations_used=0,
            last_product_response=ProductSearchResults(internal_results=[], external_results=[]),
            last_retrieval_output=_LastRetrievalOutput(internal_count=0, external_count=0),
            last_weather_context=None,
            last_retrieval_signature=None,
            retrieval_called_this_turn=False,
            retrieval_from_planner=False,
            last_planner_tool_calls=[],
            last_discovery_tools=[],
            discovery_happened_this_iteration=False,
            weather_fetched_this_turn=False,
            weather_fetched_last_turn=False,
            response_status=None,
            response_error=None,
            tool_history=[],
            iteration_history=[],
            last_generic_search_payload=None,
            last_resolved_location=None,
        )

    def begin_turn(self, turn_index: int) -> None:
        self.loop_state.begin_turn(turn_index)
        self.iterations_used = turn_index + 1
        self.weather_fetched_last_turn = self.weather_fetched_this_turn
        self.weather_fetched_this_turn = False
        self.retrieval_called_this_turn = False
        self.retrieval_from_planner = False
        self.discovery_happened_this_iteration = False
        self.last_resolved_location = None

    def add_traces(self, traces: list[ToolCallTrace]) -> None:
        for trace in traces:
            self.tool_traces.append(trace)

    def apply_decision(self, decision: dict[str, Any]) -> None:
        goal = decision.get("goal")
        if isinstance(goal, str):
            goal = goal.strip()
            if goal:
                self.loop_state.goal = goal

        done = decision.get("done", False)
        if isinstance(done, bool):
            self.loop_state.goal_reached = done

        qrr = decision.get("query_refinement_reason")
        if isinstance(qrr, str):
            qrr = qrr.strip()
            if qrr:
                self.loop_state.query_refinement_reason = qrr

    def set_response_status(self, status: str, error: str | None) -> None:
        self.response_status = status
        self.response_error = error

    def add_iteration_summary(self, summary: dict[str, Any]) -> None:
        self.iteration_history.append(summary)

    def apply_tool_execution(
        self,
        *,
        name: str,
        args: dict[str, Any],
        status: str,
        output: dict[str, Any] | None,
        error: str | None,
        turn_index: int,
        call_index: int,
        raw_result: Any = None,
    ) -> None:
        entry = {
            "name": name,
            "args": args,
            "status": status,
            "output": output,
            "error": error,
            "turn_index": turn_index,
            "call_index": call_index,
        }
        self.tool_history.append(entry)

        if status != ToolCallStatus.SUCCESS.value:
            return

        if name == ToolName.FIND_PRODUCTS.value:
            self.retrieval_called_this_turn = True
            query_text = str((args or {}).get("query_text") or "").strip()
            if query_text:
                self.loop_state.refined_query_text = query_text
                self.loop_state.current_query_text = query_text
            if isinstance(output, dict):
                internal_count = int(output.get("internal_count") or 0)
                external_count = int(output.get("external_count") or 0)
                self.last_retrieval_output = _LastRetrievalOutput(
                    internal_count=internal_count,
                    external_count=external_count,
                )
            if isinstance(raw_result, ProductSearchResults):
                self.last_product_response = raw_result

        elif name == ToolName.RESOLVE_CITY_LOCATION.value:
            if isinstance(output, dict):
                self.last_resolved_location = output

        elif name == ToolName.GET_HISTORICAL_MONTH_WEATHER.value:
            if isinstance(output, dict):
                city = str((args or {}).get("city") or "").strip()
                self.last_weather_context = {
                    "requested_city": city or None,
                    "resolved_city": (
                        (self.last_resolved_location or {}).get("name")
                        or output.get("city")
                    ),
                    "resolved_country": (
                        (self.last_resolved_location or {}).get("country")
                        or output.get("country")
                    ),
                    "year": (args or {}).get("year"),
                    "month": (args or {}).get("month"),
                    "avg_temp_max_c": output.get("avg_temp_max_c"),
                    "avg_temp_min_c": output.get("avg_temp_min_c"),
                    "total_precip_mm": output.get("total_precip_mm"),
                    "avg_wind_max_kmh": output.get("avg_wind_max_kmh"),
                }
                self.weather_fetched_this_turn = True

        elif name == ToolName.GENERIC_WEB_SEARCH.value:
            if isinstance(output, dict):
                self.last_generic_search_payload = output

    def safe_product_response(self) -> ProductSearchResults:
        if isinstance(self.last_product_response, ProductSearchResults):
            return self.last_product_response
        return ProductSearchResults(internal_results=[], external_results=[])

    def safe_retrieval_output(self) -> _LastRetrievalOutput:
        if isinstance(self.last_retrieval_output, _LastRetrievalOutput):
            return self.last_retrieval_output
        return _LastRetrievalOutput(internal_count=0, external_count=0)

    def compute_allow_web_fallback(self, turn_index: int) -> bool:
        return (
            turn_index >= MIN_INTERNAL_ONLY_TURNS_BEFORE_WEB_FALLBACK
            and not self.weather_fetched_last_turn
        )
