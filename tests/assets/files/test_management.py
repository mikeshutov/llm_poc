from capabilities.adapters.files.management import (
    _normalize_format,
    _read_document,
    _write_document,
)


def test_file_format_is_inferred_or_normalized() -> None:
    assert _normalize_format("note", "markdown") == ("note.md", "md")
    assert _normalize_format("report.docx", None) == ("report.docx", "docx")
    assert _normalize_format("report", "pdf") == ("report.pdf", "pdf")


def test_docx_documents_are_written_and_read_as_text(tmp_path) -> None:
    path = tmp_path / "joke.docx"
    _write_document(path, "Because it had a virus!", "docx", "A Short Joke")

    assert _read_document(path) == "A Short Joke\nBecause it had a virus!"
