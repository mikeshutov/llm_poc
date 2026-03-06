from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from products.models.product_search_results import ProductSearchResults
from rendering.cards_mapper import product_results_to_cards

if TYPE_CHECKING:
    from agent.agentstate.model import AgentState


@dataclass(frozen=True)
class AgentResult:
    answer: list[str]
    follow_up: str = ""
    cards: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_state(
        cls,
        *,
        answer: list[str],
        follow_up: str | None = "",
        state: AgentState,
    ) -> "AgentResult":
        cards: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for iteration in state.iteration_trace:
            for result in iteration.results.values():
                if not isinstance(result, ProductSearchResults):
                    continue

                for card in product_results_to_cards(result):
                    card_id = str(card.get("id") or "")
                    if card_id and card_id in seen_ids:
                        continue
                    if card_id:
                        seen_ids.add(card_id)
                    cards.append(card)

        return cls(
            answer=answer,
            follow_up=follow_up or "",
            cards=cards,
        )
