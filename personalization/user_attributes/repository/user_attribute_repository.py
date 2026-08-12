from __future__ import annotations

from typing import Any, Optional, Sequence
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from common.data import normalize_string_list
from db.connection import get_connection
from personalization.user_attributes.models.user_attribute_models import UserAttribute, UserAttributeSearchResult
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_VALUES
from personalization.profile.repository.repo_factory import get_user_profile_repo

ATTRIBUTE_ORDER_FIELDS = {
    "created_at": "created_at",
    "updated_at": "updated_at",
    "confidence": "confidence",
    "importance": "importance",
}
ATTRIBUTE_DUPLICATE_DISTANCE_THRESHOLD = 0.12
ATTRIBUTE_TYPES = set(ATTRIBUTE_TYPE_VALUES)


class UserAttributeRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()
        register_vector(self._conn)

    def _normalize_attribute_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)

        attribute_embedding = normalized.get("attribute_embedding")
        if attribute_embedding is not None and hasattr(attribute_embedding, "tolist"):
            normalized["attribute_embedding"] = attribute_embedding.tolist()

        for field_name in ("created_at", "updated_at"):
            field_value = normalized.get(field_name)
            if field_value is not None and hasattr(field_value, "isoformat"):
                normalized[field_name] = field_value.isoformat()

        return normalized

    def _validate_attribute_type(self, attribute_type: Optional[str]) -> None:
        if attribute_type is None:
            return
        if attribute_type not in ATTRIBUTE_TYPES:
            raise ValueError(f"Unsupported attribute_type: {attribute_type}")

    def _find_exact_attribute(
        self,
        value: Sequence[str],
        *,
        user_id: Optional[str] = None,
        attribute_type: str,
        group_key: Optional[str] = None,
        exclude_attribute_id: Optional[UUID] = None,
    ) -> Optional[UserAttribute]:
        self._validate_attribute_type(attribute_type)
        normalized_value = normalize_string_list(value)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    value,
                    NULL AS attribute_embedding,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                FROM user_attributes
                WHERE value = %s
                  AND (CAST(%s AS text) IS NULL OR user_id = %s)
                  AND (CAST(%s AS text) IS NULL OR attribute_type = %s)
                  AND group_key IS NOT DISTINCT FROM CAST(%s AS text)
                  AND (CAST(%s AS uuid) IS NULL OR id <> %s)
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (
                    normalized_value,
                    user_id,
                    user_id,
                    attribute_type,
                    attribute_type,
                    group_key,
                    exclude_attribute_id,
                    exclude_attribute_id,
                ),
            )
            row = cur.fetchone()
            return UserAttribute(**self._normalize_attribute_row(row)) if row else None

    def _find_similar_attribute(
        self,
        query_embedding: Sequence[float],
        *,
        user_id: Optional[str] = None,
        attribute_type: Optional[str] = None,
        group_key: Optional[str] = None,
        exclude_attribute_id: Optional[UUID] = None,
        distance_threshold: float = ATTRIBUTE_DUPLICATE_DISTANCE_THRESHOLD,
    ) -> Optional[UserAttributeSearchResult]:
        self._validate_attribute_type(attribute_type)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    value,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance,
                    (attribute_embedding <-> (%s)::vector) AS relevance_score
                FROM user_attributes
                WHERE attribute_embedding IS NOT NULL
                  AND (CAST(%s AS text) IS NULL OR user_id = %s)
                  AND (CAST(%s AS text) IS NULL OR attribute_type = %s)
                  AND group_key IS NOT DISTINCT FROM CAST(%s AS text)
                  AND (CAST(%s AS uuid) IS NULL OR id <> %s)
                ORDER BY attribute_embedding <-> (%s)::vector ASC
                LIMIT 1
                """,
                (
                    list(query_embedding),
                    user_id,
                    user_id,
                    attribute_type,
                    attribute_type,
                    group_key,
                    exclude_attribute_id,
                    exclude_attribute_id,
                    list(query_embedding),
                ),
            )
            row = cur.fetchone()
            if not row:
                return None
            result = UserAttributeSearchResult(**self._normalize_attribute_row(row))
            return result if result.relevance_score <= distance_threshold else None

    def _update_attribute_record(
        self,
        attribute_id: UUID,
        *,
        user_id: Optional[str] = None,
        value: Optional[Sequence[str]] = None,
        attribute_embedding: Optional[list[float]] = None,
        attribute_type: str,
        group_key: Optional[str] = None,
        source: Optional[str] = None,
        is_active: Optional[bool] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
    ) -> Optional[UserAttribute]:
        self._validate_attribute_type(attribute_type)
        normalized_value = normalize_string_list(value) if value is not None else None
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE user_attributes
                SET value = COALESCE(%s, value),
                    attribute_embedding = COALESCE((%s)::vector, attribute_embedding),
                    attribute_type = COALESCE(%s, attribute_type),
                    group_key = COALESCE(%s, group_key),
                    source = COALESCE(%s, source),
                    is_active = COALESCE(%s, is_active),
                    confidence = COALESCE(%s, confidence),
                    importance = COALESCE(%s, importance),
                    updated_at = now()
                WHERE id = %s
                  AND (CAST(%s AS text) IS NULL OR user_id = %s)
                RETURNING
                    id,
                    user_id,
                    value,
                    NULL AS attribute_embedding,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                """,
                (
                    normalized_value,
                    attribute_embedding,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    confidence,
                    importance,
                    attribute_id,
                    user_id,
                    user_id,
                ),
            )
            row = cur.fetchone()
            return UserAttribute(**self._normalize_attribute_row(row)) if row else None

    def _deactivate_attribute(self, attribute_id: UUID, *, user_id: Optional[str] = None) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_attributes
                SET is_active = false,
                    updated_at = now()
                WHERE id = %s
                  AND (CAST(%s AS text) IS NULL OR user_id = %s)
                """,
                (attribute_id, user_id, user_id),
            )

    def create_attribute(
        self,
        value: Sequence[str],
        attribute_type: str,
        user_id: Optional[str] = None,
        attribute_embedding: Optional[list[float]] = None,
        group_key: Optional[str] = None,
        source: Optional[str] = None,
        is_active: bool = True,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
    ) -> UserAttribute:
        self._validate_attribute_type(attribute_type)
        normalized_value = normalize_string_list(value)
        if user_id is not None and user_id.strip():
            get_user_profile_repo().ensure_profile(user_id)
        exact_match = self._find_exact_attribute(
            normalized_value,
            user_id=user_id,
            attribute_type=attribute_type,
            group_key=group_key,
        )
        if exact_match is not None:
            updated_attribute = self._update_attribute_record(
                exact_match.id,
                user_id=user_id,
                value=normalized_value,
                attribute_embedding=attribute_embedding,
                attribute_type=attribute_type,
                group_key=group_key,
                source=source,
                is_active=is_active,
                confidence=confidence,
                importance=importance,
            )
            assert updated_attribute is not None
            return updated_attribute

        if attribute_embedding is not None:
            similar_match = self._find_similar_attribute(
                attribute_embedding,
                user_id=user_id,
                attribute_type=attribute_type,
                group_key=group_key,
            )
            if similar_match is not None:
                updated_attribute = self._update_attribute_record(
                    similar_match.id,
                    user_id=user_id,
                    value=normalized_value,
                    attribute_embedding=attribute_embedding,
                    attribute_type=attribute_type,
                    group_key=group_key,
                    source=source,
                    is_active=is_active,
                    confidence=confidence,
                    importance=importance,
                )
                assert updated_attribute is not None
                return updated_attribute

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO user_attributes (
                    user_id,
                    value,
                    attribute_embedding,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    confidence,
                    importance
                )
                VALUES (%s, %s, (%s)::vector, %s, %s, %s, %s, %s, %s)
                RETURNING
                    id,
                    user_id,
                    value,
                    NULL AS attribute_embedding,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                """,
                (
                    user_id,
                    normalized_value,
                    attribute_embedding,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    confidence,
                    importance,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return UserAttribute(**self._normalize_attribute_row(row))

    def update_attribute(
        self,
        attribute_id: UUID,
        attribute_type: str,
        user_id: Optional[str] = None,
        value: Optional[Sequence[str]] = None,
        attribute_embedding: Optional[list[float]] = None,
        group_key: Optional[str] = None,
        source: Optional[str] = None,
        is_active: Optional[bool] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
    ) -> Optional[UserAttribute]:
        self._validate_attribute_type(attribute_type)
        normalized_value = normalize_string_list(value) if value is not None else None
        if normalized_value is not None:
            exact_match = self._find_exact_attribute(
                normalized_value,
                user_id=user_id,
                attribute_type=attribute_type,
                group_key=group_key,
                exclude_attribute_id=attribute_id,
            )
            if exact_match is not None:
                updated_attribute = self._update_attribute_record(
                    exact_match.id,
                    user_id=user_id,
                    value=normalized_value,
                    attribute_embedding=attribute_embedding,
                    attribute_type=attribute_type,
                    group_key=group_key,
                    source=source,
                    is_active=is_active if is_active is not None else True,
                    confidence=confidence,
                    importance=importance,
                )
                self._deactivate_attribute(attribute_id, user_id=user_id)
                return updated_attribute

            if attribute_embedding is not None:
                similar_match = self._find_similar_attribute(
                    attribute_embedding,
                    user_id=user_id,
                    attribute_type=attribute_type,
                    group_key=group_key,
                    exclude_attribute_id=attribute_id,
                )
                if similar_match is not None:
                    updated_attribute = self._update_attribute_record(
                        similar_match.id,
                        user_id=user_id,
                        value=normalized_value,
                        attribute_embedding=attribute_embedding,
                        attribute_type=attribute_type,
                        group_key=group_key,
                        source=source,
                        is_active=is_active if is_active is not None else True,
                        confidence=confidence,
                        importance=importance,
                    )
                    self._deactivate_attribute(attribute_id, user_id=user_id)
                    return updated_attribute

        return self._update_attribute_record(
            attribute_id,
            user_id=user_id,
            value=normalized_value,
            attribute_embedding=attribute_embedding,
            attribute_type=attribute_type,
            group_key=group_key,
            source=source,
            is_active=is_active,
            confidence=confidence,
            importance=importance,
        )

    def get_attribute(self, attribute_id: UUID) -> Optional[UserAttribute]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    value,
                    NULL AS attribute_embedding,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                FROM user_attributes
                WHERE id = %s
                """,
                (attribute_id,),
            )
            row = cur.fetchone()
            return UserAttribute(**self._normalize_attribute_row(row)) if row else None

    def list_attributes(
        self,
        *,
        limit: int = 50,
        order_by: str = "updated_at",
        descending: bool = True,
        user_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        attribute_type: Optional[str] = None,
        group_key: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[UserAttribute]:
        self._validate_attribute_type(attribute_type)
        order_field = ATTRIBUTE_ORDER_FIELDS.get(order_by)
        if order_field is None:
            raise ValueError(f"Unsupported attribute order field: {order_by}")

        order_direction = "DESC" if descending else "ASC"
        query = f"""
            SELECT
                id,
                user_id,
                value,
                NULL AS attribute_embedding,
                attribute_type,
                group_key,
                source,
                is_active,
                created_at,
                updated_at,
                confidence,
                importance
            FROM user_attributes
            WHERE (CAST(%s AS text) IS NULL OR user_id = %s)
              AND (CAST(%s AS boolean) IS NULL OR is_active = %s)
              AND (CAST(%s AS text) IS NULL OR attribute_type = %s)
              AND (CAST(%s AS text) IS NULL OR group_key = %s)
              AND (CAST(%s AS text) IS NULL OR source = %s)
            ORDER BY {order_field} {order_direction}
            LIMIT %s
        """

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                query,
                (
                    user_id,
                    user_id,
                    is_active,
                    is_active,
                    attribute_type,
                    attribute_type,
                    group_key,
                    group_key,
                    source,
                    source,
                    limit,
                ),
            )
            rows = cur.fetchall()
            return [UserAttribute(**self._normalize_attribute_row(row)) for row in rows]

    def count_attributes(
        self,
        *,
        user_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        attribute_type: Optional[str] = None,
        group_key: Optional[str] = None,
        source: Optional[str] = None,
    ) -> int:
        self._validate_attribute_type(attribute_type)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS attribute_count
                FROM user_attributes
                WHERE (CAST(%s AS text) IS NULL OR user_id = %s)
                  AND (CAST(%s AS boolean) IS NULL OR is_active = %s)
                  AND (CAST(%s AS text) IS NULL OR attribute_type = %s)
                  AND (CAST(%s AS text) IS NULL OR group_key = %s)
                  AND (CAST(%s AS text) IS NULL OR source = %s)
                """,
                (
                    user_id,
                    user_id,
                    is_active,
                    is_active,
                    attribute_type,
                    attribute_type,
                    group_key,
                    group_key,
                    source,
                    source,
                ),
            )
            row = cur.fetchone()
            return int(row["attribute_count"]) if row else 0

    def search_attributes(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        user_id: Optional[str] = None,
        is_active: Optional[bool] = True,
        attribute_type: Optional[str] = None,
        group_key: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[UserAttributeSearchResult]:
        self._validate_attribute_type(attribute_type)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    value,
                    attribute_type,
                    group_key,
                    source,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance,
                    (attribute_embedding <-> (%s)::vector) AS relevance_score
                FROM user_attributes
                WHERE attribute_embedding IS NOT NULL
                  AND (CAST(%s AS text) IS NULL OR user_id = %s)
                  AND (CAST(%s AS boolean) IS NULL OR is_active = %s)
                  AND (CAST(%s AS text) IS NULL OR attribute_type = %s)
                  AND (CAST(%s AS text) IS NULL OR group_key = %s)
                  AND (CAST(%s AS text) IS NULL OR source = %s)
                ORDER BY attribute_embedding <-> (%s)::vector ASC
                LIMIT %s
                """,
                (
                    list(query_embedding),
                    user_id,
                    user_id,
                    is_active,
                    is_active,
                    attribute_type,
                    attribute_type,
                    group_key,
                    group_key,
                    source,
                    source,
                    list(query_embedding),
                    limit,
                ),
            )
            rows = cur.fetchall()
            return [UserAttributeSearchResult(**self._normalize_attribute_row(row)) for row in rows]

