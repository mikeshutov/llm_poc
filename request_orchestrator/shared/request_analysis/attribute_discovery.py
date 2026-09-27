from __future__ import annotations

import json
import logging

from attribute_classifier.client import AttributeClassifierClient, AttributeClassifierUnavailableError
from conversation.utils import build_conversation_context_json
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_VALUES
from request_orchestrator.models.main_state import MainState

logger = logging.getLogger(__name__)


def attribute_discovery(
    main_state: MainState,
    *,
    client: AttributeClassifierClient | None = None,
) -> list[str]:
    """Select the stored user-attribute types relevant to the current request."""
    context = json.dumps(
        {
            "task": main_state.task,
            "conversation_context": json.loads(
                build_conversation_context_json(main_state.execution_context.conversation_context)
            ),
        },
        ensure_ascii=True,
    )
    classifier = client or AttributeClassifierClient()
    try:
        discovered = classifier.relevant_attributes(context, list(ATTRIBUTE_TYPE_VALUES))
    except AttributeClassifierUnavailableError:
        logger.warning("Attribute classifier service unavailable; continuing without discovered attributes")
        return []

    available = set(ATTRIBUTE_TYPE_VALUES)
    return [attribute for attribute in discovered if isinstance(attribute, str) and attribute in available]
