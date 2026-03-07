from __future__ import annotations

from pydantic import BaseModel

class SynthesisResult(BaseModel):
    result: list[str]
    follow_up: str
    clarifying_question: str = ""