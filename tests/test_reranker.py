from personalization.profile.models import UserAttributesSection, UserProfile
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from reranker import Candidate, DEFAULT_TOP_K, RerankerPrompt, rerank_candidates
from test_utilities.mock_llm import MockLLM
from uuid import uuid4


def test_rerank_candidates_skips_llm_when_candidate_count_is_at_or_below_top_k() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["candidate-2", "candidate-1", "candidate-3"]}'
    ])
    candidates = [
        Candidate(id="candidate-1", title="First"),
        Candidate(id="candidate-2", title="Second"),
        Candidate(id="candidate-3", title="Third"),
    ]

    ranked = rerank_candidates(candidates, goal="find the best option", llm=llm)

    assert [candidate.id for candidate in ranked] == ["candidate-1", "candidate-2", "candidate-3"]
    assert llm.last_prompt is None


def test_rerank_candidates_sorts_by_llm_selected_ids_when_candidate_count_exceeds_top_k() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["candidate-8", "candidate-7", "candidate-6", "candidate-5", "candidate-4", "candidate-3", "candidate-2", "candidate-1"]}'
    ])
    candidates = [Candidate(id=f"candidate-{index}", title=f"Candidate {index}") for index in range(1, 9)]

    ranked = rerank_candidates(candidates, goal="find the best option", llm=llm)

    assert [candidate.id for candidate in ranked] == [
        "candidate-8",
        "candidate-7",
        "candidate-6",
        "candidate-5",
        "candidate-4",
        "candidate-3",
    ]
    assert llm.last_prompt is not None
    assert 'Candidates (JSON):' in llm.last_prompt
    assert 'find the best option' in llm.last_prompt


def test_reranker_prompt_serializes_goal_profile_and_candidates() -> None:
    profile = UserProfile(
        user_attributes=UserAttributesSection(
            attributes=[
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=["blue"],
                    attribute_embedding=None,
                    attribute_type="style.preferences",
                    group_key=None,
                    source="explicit",
                    is_active=True,
                    created_at="2026-08-08T00:00:00Z",
                    updated_at="2026-08-08T00:00:00Z",
                    confidence=0.9,
                    importance=0.8,
                )
            ]
        )
    )
    prompt = RerankerPrompt(
        goal="find the best option",
        user_profile=profile,
        candidates=[
            Candidate(
                id="candidate-1",
                candidate_type="product",
                title="First",
                content={
                    "name": "First Name",
                    "summary": "Summary text",
                    "description": "Description text",
                    "text": "Long fallback text",
                    "url": "https://example.com",
                },
                metadata={"source": "db", "retrieval_distance": 0.2, "raw": {"debug": True}},
            )
        ],
    )

    payload = prompt.to_dict()

    assert payload["goal"] == "find the best option"
    assert payload["user_profile"]["user_attributes"]["attributes"][0]["attribute_type"] == "style.preferences"
    assert payload["candidates"][0]["id"] == "candidate-1"
    assert payload["candidates"][0]["text"] == "First Name. Summary text"
    assert "candidate_type" not in payload["candidates"][0]
    assert payload["candidates"][0]["metadata"]["source"] == "db"
    assert "raw" not in payload["candidates"][0]["metadata"]


def test_reranker_prompt_prioritizes_name_plus_summary_then_description_then_trimmed_text() -> None:
    prompt = RerankerPrompt(
        goal="rank these",
        candidates=[
            Candidate(id="candidate-1", title="First", content={"name": "Name", "summary": "Summary text", "description": "Description text"}),
            Candidate(id="candidate-2", title="Second", content={"name": "Name", "description": "Description text"}),
            Candidate(id="candidate-3", title="Third", content={"name": "Name", "text": "  very    long   text   value  "}),
            Candidate(id="candidate-4", title="Fourth", content={"summary": "Summary only"}),
        ],
    )

    payload = prompt.to_dict()

    assert payload["candidates"][0]["text"] == "Name. Summary text"
    assert payload["candidates"][1]["text"] == "Name. Description text"
    assert payload["candidates"][2]["text"] == "Name. very long text value"
    assert payload["candidates"][3]["text"] == "Summary only"


def test_reranker_prompt_renders_prompt_text() -> None:
    prompt = RerankerPrompt(
        goal="find the best option",
        candidates=[Candidate(id="candidate-1", candidate_type="product", title="First", content={"url": "https://example.com"})],
    )

    prompt_text = prompt.to_prompt_text()

    assert 'Ranking Goal:' in prompt_text
    assert 'find the best option' in prompt_text
    assert 'Candidates (JSON):' in prompt_text
    assert 'candidate-1' in prompt_text
    assert 'Response Schema:' in prompt_text
    assert 'https://example.com' not in prompt_text
    assert 'candidate_type' not in prompt_text
    assert 'ranked_ids' in prompt_text


def test_rerank_candidates_includes_user_profile_when_provided() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["candidate-8", "candidate-7", "candidate-6", "candidate-5", "candidate-4", "candidate-3", "candidate-2", "candidate-1"]}'
    ])
    profile = UserProfile(
        user_attributes=UserAttributesSection(
            attributes=[
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=["blue", "minimal"],
                    attribute_embedding=None,
                    attribute_type="style.preferences",
                    group_key=None,
                    source="explicit",
                    is_active=True,
                    created_at="2026-08-08T00:00:00Z",
                    updated_at="2026-08-08T00:00:00Z",
                    confidence=0.9,
                    importance=0.8,
                )
            ]
        )
    )
    candidates = [Candidate(id=f"candidate-{index}", title=f"Candidate {index}") for index in range(1, 9)]

    rerank_candidates(candidates, goal="find the best option", user_profile=profile, llm=llm)

    assert llm.last_prompt is not None
    assert 'User Profile (JSON):' in llm.last_prompt
    assert 'style.preferences' in llm.last_prompt
    assert 'blue' in llm.last_prompt


def test_rerank_candidates_appends_candidates_missing_from_llm_output() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["candidate-8"]}'
    ])
    candidates = [Candidate(id=f"candidate-{index}", title=f"Candidate {index}") for index in range(1, 9)]

    ranked = rerank_candidates(candidates, llm=llm)

    assert [candidate.id for candidate in ranked] == [
        "candidate-8",
        "candidate-1",
        "candidate-2",
        "candidate-3",
        "candidate-4",
        "candidate-5",
    ]


def test_rerank_candidates_falls_back_to_original_order_on_invalid_json() -> None:
    llm = MockLLM(["not valid json"])
    candidates = [Candidate(id=f"candidate-{index}", title=f"Candidate {index}") for index in range(1, 9)]

    ranked = rerank_candidates(candidates, llm=llm)

    assert [candidate.id for candidate in ranked] == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
        "candidate-4",
        "candidate-5",
        "candidate-6",
    ]


def test_rerank_candidates_returns_only_default_top_k_results() -> None:
    candidate_ids = [f"candidate-{index}" for index in range(1, 9)]
    llm = MockLLM([
        '{"ranked_ids": ["candidate-8", "candidate-7", "candidate-6", "candidate-5", "candidate-4", "candidate-3", "candidate-2", "candidate-1"]}'
    ])
    candidates = [Candidate(id=candidate_id, title=candidate_id) for candidate_id in candidate_ids]

    ranked = rerank_candidates(candidates, llm=llm)

    assert len(ranked) == DEFAULT_TOP_K
    assert [candidate.id for candidate in ranked] == [
        "candidate-8",
        "candidate-7",
        "candidate-6",
        "candidate-5",
        "candidate-4",
        "candidate-3",
    ]
