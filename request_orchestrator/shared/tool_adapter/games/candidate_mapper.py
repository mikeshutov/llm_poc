from __future__ import annotations

from typing import Any

from integrations.edhrec.models import EdhrecCardView
from reranker import RerankerCandidate, rerank_candidates


def edhrec_card_to_candidate(card: EdhrecCardView, *, section: str) -> RerankerCandidate:
    fields = [f"name: {card.name}", f"section: {section}"]
    fields.extend(f"{field}: {value}" for field, value in (
        ("synergy", card.synergy),
        ("num_decks", card.num_decks),
        ("potential_decks", card.potential_decks),
        ("trend_zscore", card.trend_zscore),
    ) if value is not None)
    return RerankerCandidate(
        id=card.id or card.slug or card.name,
        text="\n".join(fields),
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
