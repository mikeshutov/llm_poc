from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from common.serialization import prune_empty_prompt_values
from personalization.profile.models import UserProfile
from reranker.constants import DEFAULT_TOP_K, RERANKER_RESPONSE_SCHEMA
from reranker.models.candidate import Candidate


@dataclass(frozen=True)
class RerankerPrompt:
    goal: str = ""
    user_profile: UserProfile | None = None
    candidates: list[Candidate] | None = None

    def _build_candidate_text(self, candidate: Candidate) -> str | None:
        content = candidate.content or {}

        name = content.get("name")
        summary = content.get("summary")
        description = content.get("description")
        text = content.get("text")

        name_text = name.strip() if isinstance(name, str) and name.strip() else None
        summary_text = summary.strip() if isinstance(summary, str) and summary.strip() else None
        description_text = description.strip() if isinstance(description, str) and description.strip() else None
        raw_text = " ".join(text.split())[:500] if isinstance(text, str) and text.strip() else None

        if name_text and summary_text:
            return f"{name_text}. {summary_text}"
        if name_text and description_text:
            return f"{name_text}. {description_text}"
        if name_text and raw_text:
            return f"{name_text}. {raw_text}"
        if name_text:
            return name_text
        if summary_text:
            return summary_text
        if description_text:
            return description_text
        return raw_text

    def _serialize_candidate(self, candidate: Candidate) -> dict[str, Any]:
        metadata = candidate.metadata or {}

        return prune_empty_prompt_values(
            {
                "id": candidate.id,
                "title": candidate.title,
                "text": self._build_candidate_text(candidate),
                "attributes": candidate.attributes,
                "metadata": {
                    "source": metadata.get("source"),
                    "retrieval_distance": metadata.get("retrieval_distance"),
                    "flags": metadata.get("flags"),
                    "reasons": metadata.get("reasons"),
                },
            }
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "goal": self.goal.strip(),
        }

        if self.user_profile is not None:
            data["user_profile"] = self.user_profile.to_prompt_dict()

        if self.candidates is not None:
            data["candidates"] = [self._serialize_candidate(candidate) for candidate in self.candidates]

        return prune_empty_prompt_values(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, default=str)

    def to_prompt_text(self) -> str:
        payload = self.to_dict()
        goal_text = str(payload.get("goal", ""))

        parts = [
            "You are reranking candidate results for downstream selection.",
            "Return only JSON.",
            f"Rank the candidates from most relevant to least relevant and return the top {DEFAULT_TOP_K} ids first.",
            "Use the user profile when it is provided, especially any relevant stored preferences or attributes.",
            "Focus on the candidate information most useful for relevance; ignore missing fields.",
            "Preserve candidate ids exactly as provided.",
            "Do not invent ids and do not omit ids unless you are completely unable to rank them.",
        ]

        if goal_text:
            parts.extend([
                "Ranking Goal:",
                goal_text,
            ])

        if "user_profile" in payload:
            parts.extend([
                "User Profile (JSON):",
                json.dumps(payload["user_profile"], indent=2, ensure_ascii=True),
            ])

        parts.extend([
            "Candidates (JSON):",
            json.dumps(payload.get("candidates", []), indent=2, ensure_ascii=True),
            RERANKER_RESPONSE_SCHEMA,
        ])
        return "\n\n".join(parts)
