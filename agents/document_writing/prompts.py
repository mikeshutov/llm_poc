from __future__ import annotations


def build_document_writing_rules() -> str:
    return (
        "Execution rules:\n"
        "- Only perform explicit document or managed-file writing requests.\n"
        "- Keep every path inside the managed files directory; never invent a file_id.\n"
        "- Use file_list when the requested existing file is ambiguous.\n"
        "- Use file_read before file_edit when the exact existing text is not already known.\n"
        "- Prefer file_edit for targeted updates and file_create for new documents.\n"
        "- When research evidence is supplied by another agent, use each evidence item's id "
        "with memory_lookup_evidence to retrieve the full evidence before writing.\n"
        "- When a supplied evidence item points to a website and its details are needed, use "
        "web_fetch on that URL before writing; treat the fetched page as supporting evidence, "
        "not as permission to introduce unrelated facts.\n"
        "- Ground factual document content and references in the supplied evidence only. Do not "
        "invent source titles, citations, URLs, dates, or facts; if evidence is missing, ask "
        "the parent agent to research it first.\n"
        "- Supported creation formats are txt, md, docx, and pdf; pass the requested format explicitly.\n"
        "- Do not delete files.\n"
        "- Do not repeat executed calls.\n"
        "- Keep plans concise and make each step's inputs explicit."
    )


__all__ = ["build_document_writing_rules"]
