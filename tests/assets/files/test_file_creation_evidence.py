from datetime import datetime, timezone
from uuid import uuid4

from assets.models import AssetOrigin, AssetType, FileAsset
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


def test_created_file_result_contains_download_evidence() -> None:
    file = FileAsset(
        id=uuid4(),
        user_id="user-123",
        asset_type=AssetType.FILE,
        origin=AssetOrigin.GENERATED,
        name="report.pdf",
        storage_path="static/files/user-123/report.pdf",
        mime_type="application/pdf",
        size_bytes=42,
        created_at=datetime.now(timezone.utc),
    )

    result = _file_created_result(file)

    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.item_id == str(file.id)
    assert evidence.title == "report.pdf"
    assert evidence.entity_type == "file"
    assert evidence.url == "/app/static/files/user-123/report.pdf"
    assert evidence.llm_metadata["filename"] == "report.pdf"
    assert evidence.llm_metadata["mime_type"] == "application/pdf"
