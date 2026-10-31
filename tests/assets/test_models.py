from datetime import datetime, timezone
from uuid import uuid4

from assets.models import AssetOrigin, AssetReference, AssetType, FileAsset, HttpPage


def test_file_asset_carries_origin_and_file_storage_details() -> None:
    asset = FileAsset(
        id=uuid4(),
        user_id="user-1",
        origin=AssetOrigin.GENERATED,
        name="report.txt",
        storage_path="static/files/report.txt",
        mime_type="text/plain",
        created_at=datetime.now(timezone.utc),
    )

    assert asset.asset_type == AssetType.FILE
    assert asset.origin == AssetOrigin.GENERATED


def test_http_page_is_an_asset_subtype() -> None:
    page = HttpPage(
        id=uuid4(),
        user_id="user-1",
        origin=AssetOrigin.DERIVED,
        name="Example",
        url="https://example.com",
        created_at=datetime.now(timezone.utc),
    )

    assert page.asset_type == AssetType.HTTP_PAGE


def test_asset_reference_uses_canonical_asset_fields() -> None:
    reference = AssetReference(
        id=uuid4(),
        user_id="user-1",
        asset_type="file",
        asset_id="asset-123",
        created_at=datetime.now(timezone.utc),
    )

    assert reference.asset_type == "file"
    assert reference.asset_id == "asset-123"
