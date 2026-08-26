from request_orchestrator.shared.tool_adapter.user_attribute_evidence_metadata import UserAttributeEvidenceMetadata


def test_user_attribute_evidence_metadata_identifies_created_attributes() -> None:
    metadata = UserAttributeEvidenceMetadata(
        operation="created",
        group_key="favorites",
        attribute_values=["pizza", "eggs"],
    )

    assert metadata.model_dump(exclude_none=True) == {
        "operation": "created",
        "group_key": "favorites",
        "attribute_values": ["pizza", "eggs"],
    }


def test_user_attribute_evidence_metadata_identifies_updated_attributes() -> None:
    metadata = UserAttributeEvidenceMetadata(
        operation="updated",
        group_key="favorites",
        attribute_values=["pizza", "eggs"],
    )

    assert metadata.model_dump(exclude_none=True) == {
        "operation": "updated",
        "group_key": "favorites",
        "attribute_values": ["pizza", "eggs"],
    }
