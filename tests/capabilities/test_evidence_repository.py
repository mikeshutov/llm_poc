from __future__ import annotations

from uuid import uuid4

from retrieval.evidence import EvidenceView
from capabilities.registry.repository.evidence_repository import EvidenceRepository


def test_get_by_ids_reads_canonical_evidence_views(database_connection_factory) -> None:
    evidence_id = uuid4()
    row = {
        "id": evidence_id,
        "summary": "A product.",
        "source": "catalog",
        "type": "product_results",
        "data": {
            "item_id": "product-1",
            "title": "Product",
            "urls": [{"url": "https://example.com", "url_type": "website"}],
        },
        "metadata": {"price": 25.0},
    }
    connection, cursor = database_connection_factory(rows=[{"payload": {"evidence": [row]}}])
    repository = EvidenceRepository(conn=connection)

    evidence_by_id = repository.get_by_ids(
        [str(evidence_id), "not-a-uuid", str(evidence_id)],
        user_id="user-1",
    )

    assert "FROM conversation_message" in cursor.execute.call_args.args[0]
    assert "jsonb_each" not in cursor.execute.call_args.args[0]
    assert "JOIN conversation_roundtrip" in cursor.execute.call_args.args[0]
    assert "c.user_id" in cursor.execute.call_args.args[0]
    assert cursor.execute.call_args.args[1] == ("user-1", "user-1")
    assert evidence_by_id == {
        str(evidence_id): EvidenceView(
            id=evidence_id,
            summary="A product.",
            source="catalog",
        type="product_results",
        metadata={"price": 25.0},
        data={
            "item_id": "product-1",
            "title": "Product",
            "urls": [{"url": "https://example.com", "url_type": "website"}],
        })
    }


