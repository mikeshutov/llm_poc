from __future__ import annotations

from typing import Any

from integrations.edhrec.models import EdhrecCardView
from reranker import Candidate, rerank_candidates


def edhrec_card_to_candidate(card: EdhrecCardView, *, section: str) -> Candidate:
    summary_parts: list[str] = []
    if card.synergy is not None:
        summary_parts.append(f"Synergy: {card.synergy:.3f}")
    if card.num_decks:
        summary_parts.append(f"Decks: {card.num_decks}")
    if card.potential_decks:
        summary_parts.append(f"Potential decks: {card.potential_decks}")

    return Candidate(
        id=card.id or card.slug or card.name,
        title=card.name,
        content={
            "name": card.name,
            "summary": " | ".join(summary_parts) if summary_parts else None,
            "description": section,
            "url": f"https://edhrec.com{card.url}" if card.url else None,
        },
        attributes={
            "section": section,
            "synergy": card.synergy,
            "num_decks": card.num_decks,
            "potential_decks": card.potential_decks,
            "trend_zscore": card.trend_zscore,
        },
        metadata={
            "source": "edhrec",
        },
    )


def rerank_edhrec_cards(
    cards: list[tuple[str, EdhrecCardView]],
    *,
    goal: str | None = None,
    llm: Any | None = None,
    limit: int | None = None,
) -> list[tuple[str, EdhrecCardView]]:
    if not cards:
        return []

    candidates = [
        edhrec_card_to_candidate(card, section=section)
        for section, card in cards
    ]
    ranked_candidates = rerank_candidates(
        candidates,
        goal=goal,
        llm=llm,
        limit=limit,
    )
    card_by_id = {
        (card.id or card.slug or card.name): (section, card)
        for section, card in cards
    }
    ranked_cards: list[tuple[str, EdhrecCardView]] = []
    seen_ids: set[str] = set()
    for candidate in ranked_candidates:
        record = card_by_id.get(candidate.id)
        if record is None or candidate.id in seen_ids:
            continue
        ranked_cards.append(record)
        seen_ids.add(candidate.id)
    if limit is not None:
        return ranked_cards[: max(1, limit)]
    return ranked_cards
