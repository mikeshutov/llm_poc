from typing import Final, Literal

from pydantic import BaseModel, Field

USER_ATTRIBUTE_OPERATION_CREATED: Final = "created"
USER_ATTRIBUTE_OPERATION_UPDATED: Final = "updated"
USER_ATTRIBUTE_OPERATION_UPSERTED: Final = "upserted"


class UserAttributeEvidenceMetadata(BaseModel):
    operation: Literal[
        USER_ATTRIBUTE_OPERATION_CREATED,
        USER_ATTRIBUTE_OPERATION_UPDATED,
        USER_ATTRIBUTE_OPERATION_UPSERTED,
    ]
    group_key: str | None = None
    attribute_values: list[str] = Field(default_factory=list)
