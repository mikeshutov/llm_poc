from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EdhrecTagLink(BaseModel):
    count: int = 0
    slug: str = ""
    value: str = ""


class EdhrecCardView(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = ""
    name: str = ""
    sanitized: str = ""
    slug: str = ""
    url: str = ""
    synergy: float | None = None
    num_decks: int = Field(default=0, alias="num_decks")
    potential_decks: int = Field(default=0, alias="potential_decks")
    trend_zscore: float | None = Field(default=None, alias="trend_zscore")


class EdhrecComboLink(BaseModel):
    value: str = ""
    alt: str = ""
    href: str = ""


class EdhrecCardList(BaseModel):
    header: str = ""
    tag: str = ""
    cardviews: list[EdhrecCardView] = []


class EdhrecPanelCollection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    taglinks: list[EdhrecTagLink] = []
    combocounts: list[EdhrecComboLink] = []
    mana_curve: dict[str, int | float] = Field(default_factory=dict, alias="mana_curve")
    rank_over_time: dict[str, dict[str, int | float]] = Field(default_factory=dict, alias="rank_over_time")


class EdhrecJsonDict(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cardlists: list[EdhrecCardList] = []


class EdhrecContainer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    title: str = ""
    description: str = ""
    json_dict: EdhrecJsonDict = Field(default_factory=EdhrecJsonDict, alias="json_dict")


class EdhrecCommanderPage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    header: str = ""
    description: str = ""
    container: EdhrecContainer = Field(default_factory=EdhrecContainer)
    similar: list[str] = []
    budget_counts: dict[str, int] = Field(default_factory=dict, alias="budget_counts")
    bracket_counts: dict[str, int] = Field(default_factory=dict, alias="bracket_counts")
    savedate_counts: dict[str, int] = Field(default_factory=dict, alias="savedate_counts")
    panels: EdhrecPanelCollection = Field(default_factory=EdhrecPanelCollection)
    creature: int = 0
    artifact: int = 0
    enchantment: int = 0
    instant: int = 0
    sorcery: int = 0
    land: int = 0
    planeswalker: int = 0
    battle: int = 0
    basic: int = 0
    nonbasic: int = 0
