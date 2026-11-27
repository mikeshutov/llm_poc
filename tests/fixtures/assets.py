from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from assets.models import AssetOrigin, FileAsset


@pytest.fixture
def file_asset_factory():
    def create(**overrides):
        return FileAsset(**deepcopy({
            "id": uuid4(), "user_id": "user-123", "origin": AssetOrigin.GENERATED,
            "name": "report.pdf", "storage_path": "static/files/user-123/report.pdf",
            "mime_type": "application/pdf", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            **overrides,
        }))

    return create


@pytest.fixture
def writing_catalog_factory(tmp_path):
    def create(definitions=None, *, directory=None, **template_files):
        directory = tmp_path if directory is None else directory
        directory.mkdir(parents=True, exist_ok=True)
        if definitions is None:
            definitions = (
                "  - id: essay\n    name: Essay\n    description: A structured essay.\n"
                "    category: academic\n    formats: [md]\n    tags: []\n"
                "    required_fields: [title]\n    template_file: essay.md\n    version: 1\n"
            )
            template_files = {"essay.md": "{{ body }}", **template_files}
        (directory / "definitions.yaml").write_text("templates:\n" + definitions, encoding="utf-8")
        for filename, content in template_files.items():
            (directory / filename).write_text(content, encoding="utf-8")
        return directory

    return create
