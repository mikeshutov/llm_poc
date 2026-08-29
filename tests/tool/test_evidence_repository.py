from __future__ import annotations

from uuid import uuid4

from request_orchestrator.models.evidence import EvidenceView
from tool.repository.evidence_repository import EvidenceRepository


class FakeCursor:
    def __init__(self, *, fetchall_rows: list[dict]):
        self.fetchall_rows = fetchall_rows
        self.executed: list[tuple[str, tuple]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.fetchall_rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor

    def cursor(self, row_factory=None):
        return self._cursor


def test_get_by_ids_reads_canonical_evidence_views() -> None:
    evidence_id = uuid4()
    row = {
        "id": evidence_id,
        "tool_call_id": uuid4(),
        "item_id": "product-1",
        "tool_name": "find_products",
        "title": "Product",
        "summary": "A product.",
        "urls": [{"url": "https://example.com", "url_type": "website"}],
        "image_url": "",
        "published_at": "",
        "source": "catalog",
        "entity_type": "product_results",
        "location_name": "",
        "hash": "hash",
        "llm_metadata": {"price": 25.0},
        "raw_payload": {"internal_id": "product-1"},
    }
    cursor = FakeCursor(fetchall_rows=[row])
    repository = EvidenceRepository(conn=FakeConnection(cursor))

    evidence_by_id = repository.get_by_ids(
        [str(evidence_id), "not-a-uuid", str(evidence_id)],
        user_id="user-1",
    )

    assert "FROM evidence_views" in cursor.executed[0][0]
    assert "jsonb_each" not in cursor.executed[0][0]
    assert "JOIN conversation" not in cursor.executed[0][0]
    assert "evidence_views.user_id" in cursor.executed[0][0]
    assert cursor.executed[0][1] == ([evidence_id], "user-1", "user-1")
    assert evidence_by_id == {
        str(evidence_id): EvidenceView(
            id=evidence_id,
            tool_call_id=row["tool_call_id"],
            item_id="product-1",
            tool_name="find_products",
            title="Product",
            summary="A product.",
            urls=[{"url": "https://example.com", "url_type": "website"}],
            source="catalog",
            entity_type="product_results",
            hash="hash",
            llm_metadata={"price": 25.0},
            raw_payload={"internal_id": "product-1"},
        )
    }
