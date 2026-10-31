from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from assets.models import Asset, AssetOrigin, AssetReference, AssetType, FileAsset, HttpPage
from assets.repository.asset_chunk_repository import AssetChunkRepository
from db.connection import get_connection


class AssetRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    @staticmethod
    def _normalize_user_id(user_id: str | None) -> str | None:
        value = user_id.strip() if isinstance(user_id, str) else None
        return value or None

    @staticmethod
    def _file_from_row(row: dict[str, Any]) -> FileAsset:
        return FileAsset.model_validate(row)

    def create_file(
        self,
        *,
        asset_id: UUID | None = None,
        user_id: str | None = None,
        name: str,
        storage_path: str,
        mime_type: str,
        origin: AssetOrigin = AssetOrigin.GENERATED,
        metadata: dict[str, Any] | None = None,
        size_bytes: int | None = None,
        checksum: str | None = None,
    ) -> FileAsset:
        resolved_user_id = self._normalize_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO assets (id, user_id, asset_type, origin, name, metadata)
                VALUES (COALESCE(%s, gen_random_uuid()), %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    asset_type = EXCLUDED.asset_type,
                    origin = EXCLUDED.origin,
                    name = EXCLUDED.name,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                RETURNING id, user_id, asset_type, origin, name, metadata, created_at, updated_at
                """,
                (asset_id, resolved_user_id, AssetType.FILE.value, origin.value, name, Jsonb(metadata or {})),
            )
            asset = cur.fetchone()
            assert asset is not None
            cur.execute(
                """
                INSERT INTO file_assets (asset_id, storage_path, mime_type, size_bytes, checksum)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (asset_id) DO UPDATE SET
                    storage_path = EXCLUDED.storage_path,
                    mime_type = EXCLUDED.mime_type,
                    size_bytes = EXCLUDED.size_bytes,
                    checksum = EXCLUDED.checksum
                RETURNING storage_path, mime_type, size_bytes, checksum
                """,
                (asset["id"], storage_path, mime_type, size_bytes, checksum),
            )
            file_data = cur.fetchone()
            assert file_data is not None
            file_asset = self._file_from_row({**asset, **file_data})
            AssetChunkRepository(self._conn).ensure_asset_chunk(file_asset)
            return file_asset

    def delete_file(self, asset_id: UUID, *, user_id: str | None = None) -> bool:
        resolved_user_id = self._normalize_user_id(user_id)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM assets
                WHERE id = %s
                  AND (user_id IS NULL OR (%s IS NOT NULL AND user_id = %s))
                  AND asset_type = %s
                """,
                (asset_id, resolved_user_id, resolved_user_id, AssetType.FILE.value),
            )
            return cur.rowcount > 0

    def create_http_page(
        self,
        *,
        user_id: str | None = None,
        name: str,
        url: str,
        title: str = "",
        content: str = "",
        origin: AssetOrigin = AssetOrigin.DERIVED,
        metadata: dict[str, Any] | None = None,
        fetched_at: datetime | None = None,
    ) -> HttpPage:
        resolved_user_id = self._normalize_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO assets (user_id, asset_type, origin, name, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, user_id, asset_type, origin, name, metadata, created_at, updated_at
                """,
                (resolved_user_id, AssetType.HTTP_PAGE.value, origin.value, name, Jsonb(metadata or {})),
            )
            asset = cur.fetchone()
            assert asset is not None
            cur.execute(
                """
                INSERT INTO http_pages (asset_id, url, title, content, fetched_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING url, title, content, fetched_at
                """,
                (asset["id"], url, title, content, fetched_at),
            )
            page_data = cur.fetchone()
            assert page_data is not None
            page = HttpPage.model_validate({**asset, **page_data})
            chunk_source = "\n".join(
                part for part in (f"Title: {title}", f"URL: {url}", content.strip()) if part
            )
            AssetChunkRepository(self._conn).replace_content(page.id, page.asset_type, [chunk_source])
            return page

    def update_file_path(self, asset_id: UUID, *, storage_path: str, name: str, user_id: str | None = None) -> bool:
        resolved_user_id = self._normalize_user_id(user_id)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE assets a
                SET name = %s, updated_at = now()
                FROM file_assets f
                WHERE a.id = f.asset_id AND a.id = %s
                  AND (a.user_id IS NULL OR (%s IS NOT NULL AND a.user_id = %s))
                """,
                (name, asset_id, resolved_user_id, resolved_user_id),
            )
            if cur.rowcount == 0:
                return False
            cur.execute("UPDATE file_assets SET storage_path = %s WHERE asset_id = %s", (storage_path, asset_id))
            return True

    def get_file(self, asset_id: UUID, *, user_id: str | None = None) -> FileAsset | None:
        resolved_user_id = self._normalize_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT a.id, a.user_id, a.asset_type, a.origin, a.name,
                       a.metadata, a.created_at, a.updated_at,
                       f.storage_path, f.mime_type, f.size_bytes, f.checksum
                FROM assets a
                JOIN file_assets f ON f.asset_id = a.id
                WHERE a.id = %s
                  AND (a.user_id IS NULL OR (%s IS NOT NULL AND a.user_id = %s))
                  AND a.asset_type = %s
                """,
                (asset_id, resolved_user_id, resolved_user_id, AssetType.FILE.value),
            )
            row = cur.fetchone()
            return self._file_from_row(row) if row else None

    def list_files(self, user_id: str | None = None, *, query: str | None = None, limit: int = 100) -> list[FileAsset]:
        resolved_user_id = self._normalize_user_id(user_id)
        query_text = query.strip() if isinstance(query, str) else ""
        query_clause = ""
        params: list[Any] = [resolved_user_id, resolved_user_id, AssetType.FILE.value]
        if query_text:
            query_clause = " AND a.name ILIKE %s"
            params.append(f"%{query_text}%")
        params.append(limit)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT a.id, a.user_id, a.asset_type, a.origin, a.name,
                       a.metadata, a.created_at, a.updated_at,
                       f.storage_path, f.mime_type, f.size_bytes, f.checksum
                FROM assets a
                JOIN file_assets f ON f.asset_id = a.id
                WHERE (a.user_id IS NULL OR (%s IS NOT NULL AND a.user_id = %s))
                  AND a.asset_type = %s
                """ + query_clause + """
                ORDER BY a.created_at DESC
                LIMIT %s
                """,
                params,
            )
            return [self._file_from_row(row) for row in cur.fetchall()]

    def create_or_get_reference(
        self,
        *,
        user_id: str,
        asset_type: str,
        asset_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AssetReference:
        resolved_user_id = self._normalize_user_id(user_id)
        if resolved_user_id is None:
            raise ValueError("user_id is required for asset references")
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO assets (user_id, asset_type, origin, name, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, asset_type, name) DO UPDATE SET updated_at = assets.updated_at
                RETURNING id, user_id, asset_type, name, metadata, created_at
                """,
                (resolved_user_id, asset_type.strip(), AssetOrigin.GENERATED.value, asset_id.strip(), Jsonb(metadata or {})),
            )
            row = cur.fetchone()
            assert row is not None
            reference = AssetReference(
                id=row["id"],
                user_id=row["user_id"],
                asset_type=row["asset_type"],
                asset_id=row["name"],
                metadata=row["metadata"],
                created_at=row["created_at"],
            )
            AssetChunkRepository(self._conn).ensure_asset_chunk(
                Asset(
                    id=reference.id,
                    user_id=reference.user_id,
                    asset_type=reference.asset_type,
                    origin=AssetOrigin.GENERATED,
                    name=reference.asset_id,
                    metadata=reference.metadata,
                    created_at=reference.created_at,
                )
            )
            return reference
