from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SparqlResult(BaseModel):
    sparql: str
    vars: list[str] = []
    bindings: list[dict[str, Any]] = []
