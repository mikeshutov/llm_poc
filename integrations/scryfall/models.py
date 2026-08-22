from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


def _resolve_card_image_url(card: object) -> str | None:
    image_uris = getattr(card, "image_uris", None)
    if image_uris is not None:
        image_url = (
            getattr(image_uris, "normal", None)
            or getattr(image_uris, "large", None)
            or getattr(image_uris, "small", None)
            or ""
        )
        return image_url.strip() or None

    card_faces = getattr(card, "card_faces", None)
    if not card_faces:
        return None
    first_face = card_faces[0]
    first_face_image_uris = getattr(first_face, "image_uris", None)
    if first_face_image_uris is None:
        return None
    image_url = (
        getattr(first_face_image_uris, "normal", None)
        or getattr(first_face_image_uris, "large", None)
        or getattr(first_face_image_uris, "small", None)
        or ""
    )
    return image_url.strip() or None


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

    def image_url(self) -> str | None:
        return _resolve_card_image_url(self)


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


class MagicCardPriceEntry(BaseModel):
    set_name: str | None = None
    scryfall_uri: str | None = None
    image_url: str | None = None
    usd: str | None = None
    usd_foil: str | None = None
    usd_etched: str | None = None
    eur: str | None = None
    eur_foil: str | None = None
    tix: str | None = None

    @classmethod
    def from_card(cls, card: ScryfallCard) -> "MagicCardPriceEntry":
        prices = dict(card.prices or {})
        return cls(
            set_name=card.set_name,
            scryfall_uri=card.scryfall_uri or None,
            image_url=_resolve_card_image_url(card),
            usd=prices.get("usd"),
            usd_foil=prices.get("usd_foil"),
            usd_etched=prices.get("usd_etched"),
            eur=prices.get("eur"),
            eur_foil=prices.get("eur_foil"),
            tix=prices.get("tix"),
        )


class MagicCardPriceMetadataEntry(BaseModel):
    set: str
    usd: str | None = None
    usd_foil: str | None = None
    usd_etched: str | None = None
    eur: str | None = None
    eur_foil: str | None = None
    magic_online: str | None = None

    @classmethod
    def from_price_entry(cls, entry: MagicCardPriceEntry) -> "MagicCardPriceMetadataEntry":
        return cls(
            set=entry.set_name or "",
            usd=entry.usd,
            usd_foil=entry.usd_foil,
            usd_etched=entry.usd_etched,
            eur=entry.eur,
            eur_foil=entry.eur_foil,
            magic_online=entry.tix,
        )


class MagicCardPriceResult(BaseModel):
    id: str
    name: str
    scryfall_uri: str | None = None
    image_url: str | None = None
    pricing: list[MagicCardPriceEntry] = Field(default_factory=list)

    @classmethod
    def from_card(
        cls,
        card: ScryfallCard,
        *,
        pricing: list[MagicCardPriceEntry] | None = None,
    ) -> "MagicCardPriceResult":
        return cls(
            id=card.id,
            name=card.name,
            scryfall_uri=card.scryfall_uri or None,
            image_url=_resolve_card_image_url(card),
            pricing=[] if pricing is None else list(pricing),
        )
