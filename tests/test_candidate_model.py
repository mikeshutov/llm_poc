from reranker.models import Candidate, CandidateContent


def test_candidate_model_supports_simple_generic_defaults() -> None:
    candidate = Candidate(
        id="candidate-123",
        candidate_type="product",
        title="Trail Running Shoes",
        content={"summary": "Breathable daily trainer"},
        attributes={"category": "Footwear", "material": "mesh"},
    )

    assert candidate.id == "candidate-123"
    assert candidate.candidate_type == "product"
    assert candidate.title == "Trail Running Shoes"
    assert isinstance(candidate.content, CandidateContent)
    assert candidate.content.summary == "Breathable daily trainer"
    assert candidate.content["summary"] == "Breathable daily trainer"
    assert candidate.attributes["material"] == "mesh"
    assert candidate.metadata == {}
    assert candidate.embedding == []


def test_candidate_model_supports_non_product_candidates() -> None:
    candidate = Candidate(
        id="memory-456",
        candidate_type="memory",
        content={
            "text": "User said they prefer spicy vegetarian meals.",
            "created_at": "2026-08-08T10:00:00Z",
        },
        metadata={"source": "conversation_memory"},
        retrieval_rank=3,
    )

    assert candidate.candidate_type == "memory"
    assert candidate.content.text == "User said they prefer spicy vegetarian meals."
    assert candidate.content["created_at"] == "2026-08-08T10:00:00Z"
    assert candidate.metadata["source"] == "conversation_memory"
    assert candidate.retrieval_rank == 3


def test_candidate_content_supports_known_and_extra_fields() -> None:
    content = CandidateContent(
        name="Candidate name",
        description="Candidate description",
        raw_source="legacy",
    )

    assert content.name == "Candidate name"
    assert content.description == "Candidate description"
    assert content.get("raw_source") == "legacy"
    assert content["raw_source"] == "legacy"
