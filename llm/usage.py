from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable
from uuid import UUID

from common.data import sanitize_for_json_storage
from common.logging import create_conversation_event
from conversation.models.conversation_models import LlmCallRecord, LlmUsage
from conversation.repository.repo_factory import get_conversation_repo
from llm.conversation_model_config import ConversationModelConfig
from request_orchestrator.shared.runtime_context import get_current_agent_name

ONE_MILLION = Decimal("1000000")


def _get_value(payload: Any, key: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_uuid(value: str | UUID | None) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return UUID(text)
    except ValueError:
        return None


def _decimal_to_str(value: Decimal | int | float | str) -> str:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    normalized = decimal_value.normalize()
    rendered = format(normalized, 'f')
    return rendered.rstrip('0').rstrip('.') if '.' in rendered else rendered


def _normalize_llm_trace_object(value: Any) -> Any:
    normalized = sanitize_for_json_storage(value)
    if isinstance(normalized, str) and len(normalized) > 4000:
        return normalized[:4000] + "..."
    return normalized


def _normalize_latency_ms(value: Any) -> int | None:
    latency_ms = _coerce_int(value)
    if latency_ms is None:
        return None
    return max(0, latency_ms)


def _build_llm_call_metadata(
    *,
    metadata: dict[str, Any] | None,
    input_object: Any = None,
    output_object: Any = None,
    latency_ms: int | None = None,
) -> dict[str, Any]:
    payload = {} if metadata is None else dict(metadata)
    if input_object is not None:
        payload["input_object"] = _normalize_llm_trace_object(input_object)
    if output_object is not None:
        payload["output_object"] = _normalize_llm_trace_object(output_object)
    normalized_latency_ms = _normalize_latency_ms(latency_ms)
    if normalized_latency_ms is not None:
        payload["latency_ms"] = normalized_latency_ms
    return sanitize_for_json_storage(payload)


def _usage_from_openai_response(raw_response: Any) -> LlmUsage | None:
    usage = getattr(raw_response, "usage", None)
    if usage is None:
        return None

    prompt_tokens_details = _get_value(usage, "prompt_tokens_details")
    input_tokens = _coerce_int(_get_value(usage, "prompt_tokens"))
    output_tokens = _coerce_int(_get_value(usage, "completion_tokens"))
    total_tokens = _coerce_int(_get_value(usage, "total_tokens"))
    cached_input_tokens = _coerce_int(_get_value(prompt_tokens_details, "cached_tokens")) or 0
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return LlmUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _usage_from_langchain_response(raw_response: Any) -> LlmUsage | None:
    usage_metadata = getattr(raw_response, "usage_metadata", None)
    if usage_metadata is not None:
        input_token_details = _get_value(usage_metadata, "input_token_details")
        input_tokens = _coerce_int(_get_value(usage_metadata, "input_tokens"))
        output_tokens = _coerce_int(_get_value(usage_metadata, "output_tokens"))
        total_tokens = _coerce_int(_get_value(usage_metadata, "total_tokens"))
        cached_input_tokens = _coerce_int(_get_value(input_token_details, "cache_read"))
        if cached_input_tokens is None:
            cached_input_tokens = _coerce_int(_get_value(input_token_details, "cached_tokens"))
        if input_tokens is not None and output_tokens is not None:
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens
            return LlmUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_input_tokens or 0,
            )

    response_metadata = getattr(raw_response, "response_metadata", None)
    token_usage = _get_value(response_metadata, "token_usage")
    if token_usage is None:
        token_usage = _get_value(response_metadata, "usage")
    if token_usage is None:
        return None

    prompt_tokens_details = _get_value(token_usage, "prompt_tokens_details")
    input_tokens = _coerce_int(_get_value(token_usage, "prompt_tokens"))
    output_tokens = _coerce_int(_get_value(token_usage, "completion_tokens"))
    total_tokens = _coerce_int(_get_value(token_usage, "total_tokens"))
    cached_input_tokens = _coerce_int(_get_value(prompt_tokens_details, "cached_tokens")) or 0
    if input_tokens is None or output_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return LlmUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def extract_llm_usage(raw_response: Any) -> LlmUsage | None:
    return _usage_from_openai_response(raw_response) or _usage_from_langchain_response(raw_response)


def _resolve_response_model_name(raw_response: Any, fallback_model_name: str | None) -> str | None:
    response_metadata = getattr(raw_response, "response_metadata", None)
    model_name = _get_value(response_metadata, "model_name")
    if model_name:
        return str(model_name)
    return fallback_model_name


def serialize_llm_call_record(record: LlmCallRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, dict):
        metadata = dict(record.get("metadata") or {})
        input_object = record.get("input_object")
        if input_object is None:
            input_object = metadata.pop("input_object", None)
        else:
            metadata.pop("input_object", None)
        output_object = record.get("output_object")
        if output_object is None:
            output_object = metadata.pop("output_object", None)
        else:
            metadata.pop("output_object", None)
        latency_ms = _normalize_latency_ms(record.get("latency_ms"))
        if latency_ms is None:
            latency_ms = _normalize_latency_ms(metadata.pop("latency_ms", None))
        else:
            metadata.pop("latency_ms", None)
        owner_agent_name = record.get("owner_agent_name")
        if owner_agent_name is None:
            owner_agent_name = metadata.pop("owner_agent_name", None)
        else:
            metadata.pop("owner_agent_name", None)
        return {
            "agent": record.get("agent"),
            "model_scope": record.get("agent"),
            "owner_agent_name": owner_agent_name,
            "stage": record.get("stage"),
            "callsite": record.get("callsite"),
            "model": record.get("model"),
            "input_tokens": record.get("input_tokens", 0),
            "output_tokens": record.get("output_tokens", 0),
            "total_tokens": record.get("total_tokens", 0),
            "cached_input_tokens": record.get("cached_input_tokens", 0),
            "input_price_per_million_tokens": str(record.get("input_price_per_million_tokens")),
            "output_price_per_million_tokens": str(record.get("output_price_per_million_tokens")),
            "computed_input_cost": str(record.get("computed_input_cost")),
            "computed_output_cost": str(record.get("computed_output_cost")),
            "computed_total_cost": str(record.get("computed_total_cost")),
            "latency_ms": latency_ms,
            "input_object": input_object,
            "output_object": output_object,
            "metadata": metadata,
        }

    metadata = dict(record.metadata or {})
    input_object = metadata.pop("input_object", None)
    output_object = metadata.pop("output_object", None)
    latency_ms = _normalize_latency_ms(metadata.pop("latency_ms", None))
    owner_agent_name = metadata.pop("owner_agent_name", None)
    return {
        "agent": record.agent,
        "model_scope": record.agent,
        "owner_agent_name": owner_agent_name,
        "stage": record.stage,
        "callsite": record.callsite,
        "model": record.model,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "cached_input_tokens": record.cached_input_tokens,
        "input_price_per_million_tokens": _decimal_to_str(record.input_price_per_million_tokens),
        "output_price_per_million_tokens": _decimal_to_str(record.output_price_per_million_tokens),
        "computed_input_cost": _decimal_to_str(record.computed_input_cost),
        "computed_output_cost": _decimal_to_str(record.computed_output_cost),
        "computed_total_cost": _decimal_to_str(record.computed_total_cost),
        "latency_ms": latency_ms,
        "input_object": input_object,
        "output_object": output_object,
        "metadata": metadata,
    }


def build_llm_usage_payload(records: Iterable[LlmCallRecord]) -> dict[str, Any]:
    serialized_calls = [serialize_llm_call_record(record) for record in records]
    total_input_tokens = sum(int(call.get("input_tokens") or 0) for call in serialized_calls)
    total_cached_input_tokens = sum(int(call.get("cached_input_tokens") or 0) for call in serialized_calls)
    total_output_tokens = sum(int(call.get("output_tokens") or 0) for call in serialized_calls)
    total_tokens = sum(int(call.get("total_tokens") or 0) for call in serialized_calls)
    total_latency_ms = sum(int(call.get("latency_ms") or 0) for call in serialized_calls)
    total_input_cost = sum(Decimal(str(call.get("computed_input_cost") or "0")) for call in serialized_calls)
    total_output_cost = sum(Decimal(str(call.get("computed_output_cost") or "0")) for call in serialized_calls)
    total_cost = total_input_cost + total_output_cost

    return {
        "retrieved_call_count": len(serialized_calls),
        "summary": {
            "input_tokens": total_input_tokens,
            "cached_input_tokens": total_cached_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency_ms,
            "computed_input_cost": _decimal_to_str(total_input_cost),
            "computed_output_cost": _decimal_to_str(total_output_cost),
            "computed_total_cost": _decimal_to_str(total_cost),
        },
        "calls": serialized_calls,
    }


def record_llm_call(
    *,
    raw_response: Any,
    model_name: str | None,
    provider: str | None = None,
    conversation_id: str | UUID | None,
    roundtrip_id: str | UUID | None,
    user_id: str | None,
    callsite: str,
    agent: str | None = None,
    stage: str | None = None,
    metadata: dict[str, Any] | None = None,
    input_object: Any = None,
    output_object: Any = None,
    latency_ms: int | None = None,
    owner_agent_name: str | None = None,
):
    usage = extract_llm_usage(raw_response)
    if usage is None:
        return None

    resolved_model_name = _resolve_response_model_name(raw_response, model_name)
    if not resolved_model_name:
        return None

    if provider is None:
        pricing = ConversationModelConfig.resolve_model_pricing(resolved_model_name)
    else:
        pricing = ConversationModelConfig.resolve_model_pricing(provider, resolved_model_name)
    input_cost = (Decimal(usage.input_tokens) * pricing.input_price_per_million_tokens) / ONE_MILLION
    output_cost = (Decimal(usage.output_tokens) * pricing.output_price_per_million_tokens) / ONE_MILLION
    total_cost = input_cost + output_cost

    repo = get_conversation_repo()
    if not hasattr(repo, "create_llm_call"):
        return None

    resolved_owner_agent_name = (owner_agent_name or get_current_agent_name() or "").strip() or None
    metadata_with_owner = {} if metadata is None else dict(metadata)
    if resolved_owner_agent_name:
        metadata_with_owner["owner_agent_name"] = resolved_owner_agent_name

    record = repo.create_llm_call(
        conversation_id=_parse_uuid(conversation_id),
        roundtrip_id=_parse_uuid(roundtrip_id),
        user_id=user_id,
        agent=agent,
        stage=stage,
        callsite=callsite,
        model=resolved_model_name,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        input_price_per_million_tokens=pricing.input_price_per_million_tokens,
        output_price_per_million_tokens=pricing.output_price_per_million_tokens,
        computed_input_cost=input_cost,
        computed_output_cost=output_cost,
        computed_total_cost=total_cost,
        metadata=_build_llm_call_metadata(
            metadata=metadata_with_owner,
            input_object=input_object,
            output_object=output_object,
            latency_ms=latency_ms,
        ),
    )
    serialized_record = serialize_llm_call_record(record)
    create_conversation_event(
        event_type="llm_call",
        source=callsite,
        agent_name=resolved_owner_agent_name or agent or "",
        node_name=stage or "",
        payload=serialized_record,
    )
    return record
