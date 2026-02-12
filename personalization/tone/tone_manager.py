from typing import Optional

from personalization.tone.models import Tone


def _coerce_tone(value: Optional[Tone | dict]) -> Optional[Tone]:
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


def update_tone_state(previous: Optional[Tone], current: Optional[Tone]) -> Optional[Tone]:
    previous = _coerce_tone(previous)
    current = _coerce_tone(current)
    if current is None:
        return previous
    if previous is None:
        return current
    if current.override:
        return current
    if current.label == previous.label:
        blended = 0.7 * previous.score + 0.3 * current.score
        return Tone(label=previous.label, score=blended, override=False)
    # If the new tone is confident enough, switch; otherwise keep previous.
    if current.score >= 0.7:
        return Tone(label=current.label, score=current.score, override=False)
    blended = 0.7 * previous.score + 0.3 * current.score
    return Tone(label=previous.label, score=blended, override=False)
