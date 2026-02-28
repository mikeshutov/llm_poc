import json
from dataclasses import asdict, dataclass
from typing import Any

from agent.models.tool_call_status import ToolCallStatus
from agent.runtime.tool_executor import execute_tool
from agent.state import _AgentState, _LastRetrievalOutput
from agent.tools.tool_name import ToolName
from products.models.product_search_results import ProductSearchResults
from tool_registry import call_tool
from websearch.domain.product_retrieval import get_last_fallback_meta
from websearch.models.common_properties import CommonProperties


@dataclass(frozen=True)
class DispatchResult:
    status: str
    output_text: str
    output_payload: dict[str, Any] | None
    error_message: str | None
    duration_ms: int
    raw_result: Any = None


def _coerce_common_filters(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    try:
        return CommonProperties(**payload).model_dump()
    except Exception:
        return None


def _coerce_product_filters(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    product_filters: dict[str, Any] = {}

    style = payload.get("style")
    if style is not None:
        style_text = str(style).strip()
        if style_text:
            product_filters["style"] = style_text

    category = payload.get("category")
    if category is not None:
        category_text = str(category).strip()
        if category_text:
            product_filters["category"] = category_text

    return product_filters or None


def _signature(
    query_text: str,
    common_filters: dict[str, Any] | None,
    product_filters: dict[str, Any] | None,
    fallback_mode: str,
) -> str:
    return json.dumps(
        {
            "query_text": query_text.strip(),
            "common_filters": common_filters or {},
            "product_filters": product_filters or {},
            "runtime_fallback_mode": fallback_mode,
        },
        sort_keys=True,
    )


class ToolDispatcher:
    def __init__(
        self,
        *,
        allowed_tools: set[ToolName],
        state: _AgentState,
        allow_web_fallback: bool,
    ) -> None:
        self._state = state
        self._allow_web_fallback = allow_web_fallback
        self._allowed_values = {tool.value for tool in allowed_tools}

    def dispatch(self, name: str, args: dict[str, Any]) -> DispatchResult:
        if name not in {t.value for t in ToolName} or name not in self._allowed_values:
            return DispatchResult(
                status=ToolCallStatus.BLOCKED.value,
                output_text="blocked_by_policy",
                output_payload=None,
                error_message="Tool is not allowed for this route.",
                duration_ms=0,
            )

        if name == ToolName.FIND_PRODUCTS.value:
            query_text = str(args.get("query_text") or self._state.loop_state.refined_query_text).strip()
            common_filters = _coerce_common_filters(args.get("common_filters"))
            product_filters = _coerce_product_filters(args.get("product_filters"))
            runtime_fallback_mode = "eligible_for_fallback" if self._allow_web_fallback else "internal_only"
            sig = _signature(query_text, common_filters, product_filters, runtime_fallback_mode)
            if sig == self._state.last_retrieval_signature:
                return DispatchResult(
                    status=ToolCallStatus.SKIPPED.value,
                    output_text="dedupe_skipped",
                    output_payload={
                        "internal_count": self._state.last_retrieval_output.internal_count,
                        "external_count": self._state.last_retrieval_output.external_count,
                        "dedupe_skipped": True,
                        "runtime_fallback_mode": runtime_fallback_mode,
                        "signature": sig,
                    },
                    error_message=None,
                    duration_ms=0,
                )

            status = ToolCallStatus.SUCCESS.value
            error_message: str | None = None
            output_payload: dict[str, Any] | None = None
            raw_result: ProductSearchResults | None = None
            duration_ms = 0
            try:
                raw_result = call_tool(
                    ToolName.FIND_PRODUCTS.value,
                    {
                        "query_text": query_text,
                        "common_filters": common_filters,
                        "product_filters": product_filters,
                        "allow_web_fallback": self._allow_web_fallback,
                    },
                )
                retrieval_output = _LastRetrievalOutput(
                    internal_count=len(raw_result.internal_results),
                    external_count=len(raw_result.external_results),
                )
                self._state.last_retrieval_signature = sig
                output_payload = {
                    **asdict(retrieval_output),
                    "internal_search_executed": True,
                    "original_query_text": self._state.loop_state.original_query_text,
                    "refined_query_text": self._state.loop_state.refined_query_text,
                    "query_used_for_retrieval": query_text,
                    "runtime_fallback_mode": runtime_fallback_mode,
                    "dedupe_skipped": False,
                    "web_fallback_allowed": self._allow_web_fallback,
                    "web_fallback_eligible": retrieval_output.internal_count == 0,
                    "web_fallback_used": (
                        self._allow_web_fallback
                        and retrieval_output.internal_count == 0
                        and retrieval_output.external_count > 0
                    ),
                    "signature": sig,
                    **get_last_fallback_meta(),
                }
            except Exception as exc:
                status = ToolCallStatus.ERROR.value
                error_message = str(exc)

            return DispatchResult(
                status=status,
                output_text="products_retrieval",
                output_payload=output_payload,
                error_message=error_message,
                duration_ms=duration_ms,
                raw_result=raw_result,
            )

        status, output_payload, error_message, duration_ms = execute_tool(name, args)
        return DispatchResult(
            status=status,
            output_text="tool_executed",
            output_payload=output_payload,
            error_message=error_message,
            duration_ms=duration_ms,
        )
