from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ResponsePayload:
    response: str
    cards: List[Dict[str, Any]]
    follow_up: str
