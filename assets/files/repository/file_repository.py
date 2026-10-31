from __future__ import annotations

from typing import Any
from uuid import UUID

from assets.models import AssetOrigin, FileAsset
from assets.repository.asset_repository import AssetRepository
from db.connection import get_connection


class FileRepository:
    """File-focused facade over the canonical asset repository."""

    def __init__(self) -> None:
        self._assets = AssetRepository(get_connection())

    def get_file_by_id(self, file_id: UUID, *, user_id: str | None = None) -> FileAsset | None:
        return self._assets.get_file(file_id, user_id=user_id)

    def list_files(self, *, user_id: str | None = None, query: str | None = None) -> list[FileAsset]:
        return self._assets.list_files(user_id, query=query)

    def delete_file(self, file_id: UUID, *, user_id: str | None = None) -> FileAsset | None:
        file = self._assets.get_file(file_id, user_id=user_id)
        if file is None or not self._assets.delete_file(file_id, user_id=user_id):
            return None
        return file

    def update_file_path(
        self,
        file_id: UUID,
        file_path: str,
        file_name: str,
        *,
        user_id: str | None = None,
    ) -> FileAsset | None:
        file = self._assets.get_file(file_id, user_id=user_id)
        if file is None or not self._assets.update_file_path(
            file_id,
            storage_path=file_path,
            name=file_name,
            user_id=user_id,
        ):
            return None
        return self._assets.get_file(file_id, user_id=user_id)

    def create_file(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        metadata: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
        conversation_id: UUID | None = None,
        origin: AssetOrigin = AssetOrigin.UPLOADED,
    ) -> FileAsset:
        asset_metadata = dict(metadata or {})
        if conversation_id is not None:
            asset_metadata.setdefault("conversation_id", str(conversation_id))
        return self._assets.create_file(
            user_id=user_id,
            name=file_name,
            storage_path=file_path,
            mime_type=file_type,
            origin=origin,
            metadata=asset_metadata,
        )
