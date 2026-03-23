from __future__ import annotations

from agents.impl.file_parser.state import FileParserState
from agents.graph_constants import PARSE_EDGE, VALIDATE_EDGE


def validate_parse(state: FileParserState) -> FileParserState:
    # TODO: check for missing fields and populate state.missing_fields
    state.valid = True
    return state


def validation_router(state: FileParserState) -> str:
    if not state.valid:
        return PARSE_EDGE
    return VALIDATE_EDGE
