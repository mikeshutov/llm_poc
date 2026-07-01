from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from integrations.brave.models import NewsSearchResponse
from products.models.product_search_results import ProductSearchResults
from rendering.cards_mapper import news_response_to_cards, product_results_to_cards

if TYPE_CHECKING:
    from agent.agentstate.model import AgentState


@dataclass(frozen=True)
class AgentResult:
    answer: list[str]
    follow_up: str = ""
    clarifying_question: str = ""
    roundtrip_summary: str = ""
    tool_summary: dict[str, Any] = field(default_factory=dict)
    cards: list[dict[str, Any]] = field(default_factory=list)

    @property
    def raw_response(self) -> str:
        return "\n\n".join(p for p in self.answer if p)

    def to_payload_for_update_roundtrip(self) -> dict[str, Any]:
        return {
            "response": self.raw_response,
            "cards": self.cards,
            "follow_up": self.follow_up,
            "clarifying_question": self.clarifying_question,
            "roundtrip_summary": self.roundtrip_summary,
            "tool_summary": self.tool_summary,
        }

    @classmethod
    def from_state(
        cls,
        *,
        answer: list[str],
        follow_up: str | None = "",
        clarifying_question: str | None = "",
        roundtrip_summary: str | None = "",
        tool_summary: dict[str, Any] | None = None,
        state: AgentState,
    ) -> "AgentResult":
        cards: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for iteration in state.iteration_trace:
            for result in iteration.results.values():
                if isinstance(result, ProductSearchResults):
                    new_cards = product_results_to_cards(result)
                elif isinstance(result, NewsSearchResponse):
                    new_cards = news_response_to_cards(result)
                else:
                    continue

                for card in new_cards:
                    card_id = str(card.get("id") or "")
                    if card_id and card_id in seen_ids:
                        continue
                    if card_id:
                        seen_ids.add(card_id)
                    cards.append(card)

        return cls(
            answer=answer,
            follow_up=follow_up or "",
            clarifying_question=clarifying_question or "",
            roundtrip_summary=roundtrip_summary or "",
            tool_summary=tool_summary or {},
            cards=cards,
        )
