from __future__ import annotations

from typing import Any
from uuid import UUID

from personalization.tone.models import Tone
from personalization.tone.tone_manager import update_tone_state


def _coerce_tone(value: Tone | dict[str, Any] | None) -> Tone | None:
    if value is None:
        return None
    if isinstance(value, Tone):
        return value
    if isinstance(value, dict):
        try:
            return Tone(**value)
        except Exception:
            return None
    return None


def resolve_conversation_tone_state(
    conversation_repository: Any,
    conversation_id: UUID,
    parsed_tone: Tone | dict[str, Any] | None,
) -> Tone | None:
    previous_tone: Tone | None = None
    try:
        conversation = conversation_repository.get_conversation(conversation_id)
        previous_tone = _coerce_tone(conversation.tone_state if conversation else None)
    except Exception:
        previous_tone = None

    next_tone = update_tone_state(previous_tone, _coerce_tone(parsed_tone))
    if next_tone is not None:
        try:
            if previous_tone != next_tone:
                conversation_repository.update_tone_state(conversation_id, next_tone.model_dump())
        except Exception:
            # Tone persistence should not block the response flow.
            pass
    return next_tone


def resolve_conversation_tone_label(
    conversation_repository: Any,
    conversation_id: UUID,
    parsed_tone: Tone | dict[str, Any] | None,
) -> str | None:
    tone_state = resolve_conversation_tone_state(
        conversation_repository=conversation_repository,
        conversation_id=conversation_id,
        parsed_tone=parsed_tone,
    )
    return tone_state.label.value if tone_state else None
