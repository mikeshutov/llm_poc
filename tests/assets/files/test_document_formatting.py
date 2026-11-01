from capabilities.adapters.files.management import _document_paragraphs


def test_document_paragraphs_use_blank_lines_as_boundaries_without_rewriting_text() -> None:
    assert _document_paragraphs("First line\ncontinues.\n\nSecond paragraph.") == [
        "First line continues.",
        "Second paragraph.",
    ]
