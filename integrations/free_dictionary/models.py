from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Phonetic(BaseModel):
    text: Optional[str] = None
    audio: Optional[str] = None


class Definition(BaseModel):
    definition: str
    example: Optional[str] = None
    synonyms: list[str] = []
    antonyms: list[str] = []


class Meaning(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    part_of_speech: Optional[str] = Field(default=None, alias="partOfSpeech")
    definitions: list[Definition] = []
    synonyms: list[str] = []
    antonyms: list[str] = []


class DictionaryEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    word: str
    phonetic: Optional[str] = None
    phonetics: list[Phonetic] = []
    meanings: list[Meaning] = []
    source_urls: list[str] = Field(default=[], alias="sourceUrls")
