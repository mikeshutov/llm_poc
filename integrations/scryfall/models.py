from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScryfallImageUris(BaseModel):
    small: str | None = None
    normal: str | None = None
    large: str | None = None
    png: str | None = None
    art_crop: str | None = Field(default=None, alias="art_crop")
    border_crop: str | None = Field(default=None, alias="border_crop")


class ScryfallCardFace(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = ""
    mana_cost: str | None = Field(default=None, alias="mana_cost")
    type_line: str | None = Field(default=None, alias="type_line")
    oracle_text: str | None = Field(default=None, alias="oracle_text")
    image_uris: ScryfallImageUris | None = Field(default=None, alias="image_uris")
    colors: list[str] | None = None


class ScryfallCard(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    lang: str = ""
    mana_cost: str | None = Field(default=None, alias="mana_cost")
    cmc: float | None = None
    type_line: str | None = Field(default=None, alias="type_line")
    oracle_text: str | None = Field(default=None, alias="oracle_text")
    colors: list[str] | None = None
    color_identity: list[str] | None = Field(default=None, alias="color_identity")
    image_uris: ScryfallImageUris | None = Field(default=None, alias="image_uris")
    card_faces: list[ScryfallCardFace] | None = Field(default=None, alias="card_faces")
    scryfall_uri: str = Field(default="", alias="scryfall_uri")
    set_name: str | None = Field(default=None, alias="set_name")
    rarity: str | None = None
    prices: dict[str, str | None] = {}
    legalities: dict[str, str] = {}
    games: list[str] = []


class ScryfallCardSearchResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    object: str = "list"
    total_cards: int = Field(default=0, alias="total_cards")
    has_more: bool = Field(default=False, alias="has_more")
    data: list[ScryfallCard] = []
    warnings: list[str] = []


class ScryfallRuling(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    oracle_id: str | None = Field(default=None, alias="oracle_id")
    source: str = ""
    published_at: str = Field(default="", alias="published_at")
    comment: str = ""


class ScryfallRulingList(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    object: str = "list"
    has_more: bool = Field(default=False, alias="has_more")
    data: list[ScryfallRuling] = []
