from request_orchestrator.shared.tool_adapter.user_attribute_evidence_metadata import (
    USER_ATTRIBUTE_OPERATION_CREATED,
    USER_ATTRIBUTE_OPERATION_UPDATED,
    UserAttributeEvidenceMetadata,
)


def test_user_attribute_evidence_metadata_identifies_created_attributes() -> None:
    metadata = UserAttributeEvidenceMetadata(
        operation=USER_ATTRIBUTE_OPERATION_CREATED,
        group_key="favorites",
        attribute_values=["pizza", "eggs"],
    )

    assert metadata.model_dump(exclude_none=True) == {
        "operation": USER_ATTRIBUTE_OPERATION_CREATED,
        "group_key": "favorites",
        "attribute_values": ["pizza", "eggs"],
    }


def test_user_attribute_evidence_metadata_identifies_updated_attributes() -> None:
    metadata = UserAttributeEvidenceMetadata(
        operation=USER_ATTRIBUTE_OPERATION_UPDATED,
        group_key="favorites",
        attribute_values=["pizza", "eggs"],
    )

    assert metadata.model_dump(exclude_none=True) == {
        "operation": USER_ATTRIBUTE_OPERATION_UPDATED,
        "group_key": "favorites",
        "attribute_values": ["pizza", "eggs"],
    }
