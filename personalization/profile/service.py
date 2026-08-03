from __future__ import annotations

from personalization.profile.models import GeoMetadata, UserAttributesSection, UserProfile
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo

# simple build function for now we grab everything to assemble the profile.
def build_user_profile(*, geometadata: GeoMetadata | None = None, attribute_limit: int = 100) -> UserProfile:
    attributes = get_user_attribute_repo().list_attributes(limit=attribute_limit, is_active=True)
    return UserProfile(
        geometadata=geometadata,
        user_attributes=UserAttributesSection(attributes=attributes),
    )
