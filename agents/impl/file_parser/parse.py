from __future__ import annotations

import json

from agents.impl.file_parser.state import FileParserState
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_SYSTEM, ROLE_USER
from common.parsing import strip_code_fences

SYSTEM_PROMPT = """You are a document parser. Extract the key structured information from the provided document content.
Return a JSON object with the relevant fields based on the document type.
For a resume include: name, contact, summary, skills, experience, education.
For a job description include: title, company, requirements, responsibilities, qualifications.
For other documents extract whatever structured fields are most relevant.
Return only valid JSON."""


def parse_file(state: FileParserState) -> FileParserState:
    prompt = [
        {ROLE_KEY: ROLE_SYSTEM, CONTENT_KEY: SYSTEM_PROMPT},
        {ROLE_KEY: ROLE_USER, CONTENT_KEY: f"File: {state.file_name}\n\n{state.raw_content}"},
    ]
    raw = strip_code_fences(state.llm.invoke(prompt).content)

    try:
        state.extracted = json.loads(raw)
    except Exception:
        state.extracted = {"raw": state.raw_content}

    state.done = True
    return state
