from __future__ import annotations


def build_document_writing_rules() -> str:
    return (
        "Execution rules:\n"
        "- Only perform explicit document or managed-file writing requests.\n"
        "- Use managed file tools only to discover or read an existing source document; never "
        "invent a file_id.\n"
        "- Draft and revise every document in the current conversation workspace using "
        "workspace_file_write, workspace_file_read, workspace_file_edit, and workspace_file_list.\n"
        "- Use workspace_file_promote only after the document is complete and validated; promotion "
        "copies it into managed artifacts while preserving the workspace draft.\n"
        "- For every new document, select a writing template before calling workspace_file_write unless "
        "the user or parent agent explicitly supplied the template or an equivalent required "
        "section structure. Call writing_templates with the requested format and closest "
        "category when no template was supplied; then choose the best matching template.\n"
        "- Use the selected template's required_fields to structure the document, including all "
        "applicable required sections. Never silently create a free-form document when a catalog "
        "template is available; if no suitable template exists, report that to the parent agent "
        "instead of inventing a structure.\n"
        "- When research evidence is supplied by another agent, use each evidence item's id "
        "with memory_lookup_evidence to retrieve the full evidence before writing.\n"
        "- When a supplied evidence item points to a website and its details are needed, use "
        "web_fetch on that URL before writing; treat the fetched page as supporting evidence, "
        "not as permission to introduce unrelated facts.\n"
        "- Ground factual document content and references in the supplied evidence only. Do not "
        "invent source titles, citations, URLs, dates, or facts; if evidence is missing, ask "
        "the parent agent to research it first.\n"
        "- Supported creation formats are txt, md, docx, and pdf; pass the requested format explicitly "
        "to workspace_file_write.\n"
        "- Do not delete files.\n"
        "- Do not repeat executed calls.\n"
        "- Keep plans concise and make each step's inputs explicit."
    )


__all__ = [
    "build_document_writing_rules",
]
