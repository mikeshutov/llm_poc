from __future__ import annotations

from dataclasses import replace

from personalization.profile.models import GeoMetadata, UserAttributesSection, UserProfile
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_VALUES
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo

VALID_ATTRIBUTE_TYPES = set(ATTRIBUTE_TYPE_VALUES)


def _condense_attributes(attributes: list[UserAttribute]) -> list[UserAttribute]:
    condensed_by_key: dict[tuple[str, str | None], UserAttribute] = {}

    for attribute in attributes:
        key = (attribute.attribute_type, attribute.group_key)
        existing = condensed_by_key.get(key)
        if existing is None:
            condensed_by_key[key] = replace(attribute, value=list(attribute.value))
            continue

        merged_values = list(existing.value)
        seen_values = set(merged_values)
        for value in attribute.value:
            if value not in seen_values:
                seen_values.add(value)
                merged_values.append(value)

        condensed_by_key[key] = replace(existing, value=merged_values)

    return list(condensed_by_key.values())


def build_user_profile(
    *,
    user_id: str | None = None,
    geometadata: GeoMetadata | None = None,
    attributes: list[UserAttribute] | None = None,
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        geometadata=geometadata,
        user_attributes=UserAttributesSection(attributes=[] if attributes is None else attributes),
    )


def load_user_profile_attributes(
    user_profile: UserProfile,
    *,
    requested_attribute_types: list[str],
    attribute_limit: int = 100,
) -> UserProfile:
    normalized_types: list[str] = []
    for attribute_type in requested_attribute_types:
        normalized_type = str(attribute_type).strip()
        if normalized_type and normalized_type in VALID_ATTRIBUTE_TYPES and normalized_type not in normalized_types:
            normalized_types.append(normalized_type)

    if not normalized_types:
        user_profile.user_attributes = UserAttributesSection(attributes=[])
        return user_profile

    repo = get_user_attribute_repo()
    loaded_attributes: list[UserAttribute] = []
    seen_attribute_ids: set[str] = set()

    per_type_limit = max(1, attribute_limit)
    for attribute_type in normalized_types:
        attributes = repo.list_attributes(
            limit=per_type_limit,
            user_id=user_profile.user_id,
            is_active=True,
            attribute_type=attribute_type,
        )
        for attribute in attributes:
            attribute_id = str(attribute.id)
            if attribute_id in seen_attribute_ids:
                continue
            seen_attribute_ids.add(attribute_id)
            loaded_attributes.append(attribute)
            if len(loaded_attributes) >= attribute_limit:
                user_profile.user_attributes = UserAttributesSection(
                    attributes=_condense_attributes(loaded_attributes)
                )
                return user_profile

    user_profile.user_attributes = UserAttributesSection(
        attributes=_condense_attributes(loaded_attributes)
    )
    return user_profile
