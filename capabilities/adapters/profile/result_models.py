from __future__ import annotations

from pydantic import BaseModel


class UserProfileUpdateResult(BaseModel):
    user_id: str | None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
