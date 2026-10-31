from __future__ import annotations

from time import perf_counter

from serviceclients.attribute_classifier import AttributeClassifierClient, AttributeClassifierUnavailableError
from personalization.user_attributes.models.user_attribute_types import (
    ATTRIBUTE_TYPE_VALUES,
)
from agents.discovery.agent_selection_query import build_agent_selection_query
from agent_runtime.models.main_state import MainState
from common.logging import record_node_log
from agent_runtime.constants import NodeName

MAX_DISCOVERED_ATTRIBUTE_TYPES = 10


def attribute_discovery(
    main_state: MainState,
    *,
    client: AttributeClassifierClient | None = None,
) -> list[str]:
    """Select the stored user-attribute types relevant to the current request."""
    context = main_state.discovery_query or build_agent_selection_query(
        current_user_request=main_state.task,
        conversation_context=main_state.execution_context.conversation_context,
    )
    classifier = client or AttributeClassifierClient()
    started_at = perf_counter()
    try:
        discovered = classifier.relevant_attributes(context, list(ATTRIBUTE_TYPE_VALUES))
    except AttributeClassifierUnavailableError:
        record_node_log(
            node_name=NodeName.ATTRIBUTE_DISCOVERY,
            stage="enrich_state",
            event_type="attribute_discovery",
            roundtrip_id=main_state.execution_context.roundtrip_id,
            level="warning",
            message="Attribute classifier unavailable",
            payload={"query": context, "discovered_attribute_types": []},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        return []
    available = set(ATTRIBUTE_TYPE_VALUES)
    result = []
    for attribute in discovered:
        if not isinstance(attribute, str):
            continue
        normalized = attribute.strip()
        if normalized in available and normalized not in result:
            result.append(normalized)
        if len(result) >= MAX_DISCOVERED_ATTRIBUTE_TYPES:
            break
    record_node_log(
        node_name=NodeName.ATTRIBUTE_DISCOVERY,
        stage="enrich_state",
        event_type="attribute_discovery",
        roundtrip_id=main_state.execution_context.roundtrip_id,
        message="Attribute discovery completed",
        payload={
            "query": context,
            "attribute_types_considered": list(ATTRIBUTE_TYPE_VALUES),
            "discovered_attribute_types": result,
        },
        duration_ms=int((perf_counter() - started_at) * 1000),
    )
    return result
