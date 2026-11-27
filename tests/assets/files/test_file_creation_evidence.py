from assets.files.writer import normalize_file_format as _normalize_format
from capabilities.adapters.files.management import _file_created_result


def test_file_format_normalization_accepts_descriptive_and_mime_values() -> None:
    assert _normalize_format("report", "Microsoft Word document (.docx)") == ("report.docx", "docx")
    assert _normalize_format("report", "application/pdf") == ("report.pdf", "pdf")
    assert _normalize_format("report.txt", "text/plain") == ("report.txt", "txt")
    assert _normalize_format("report.doc", None) == ("report.docx", "docx")
    assert _normalize_format("report.txt", "doc") == ("report.docx", "docx")


def test_file_format_normalization_supports_svg() -> None:
    assert _normalize_format("chart", "image/svg+xml") == ("chart.svg", "svg")


def test_created_file_result_contains_download_evidence(file_asset_factory) -> None:
    file = file_asset_factory(size_bytes=42)

    result = _file_created_result(file)

    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.data["item_id"] == str(file.id)
    assert evidence.data["title"] == "report.pdf"
    assert evidence.type == "file"
    assert evidence.data["urls"][0]["url"] == "/api/files/user-123/report.pdf"
    assert evidence.metadata["filename"] == "report.pdf"
    assert evidence.metadata["mime_type"] == "application/pdf"
