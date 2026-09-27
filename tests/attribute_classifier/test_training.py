from __future__ import annotations

from services.attribute_classifier.training.train import build_binary_examples, load_source_examples


def test_seed_data_uses_the_manual_label_format() -> None:
    from pathlib import Path

    path = Path("services/attribute_classifier/training/training_data.jsonl")
    examples = load_source_examples(path)
    assert examples
    assert all(isinstance(example["text"], str) for example in examples)
    assert all(isinstance(example["labels"], list) for example in examples)


def test_source_examples_expand_to_binary_attribute_examples() -> None:
    rows = build_binary_examples([{"text": "I like pasta", "labels": ["food.likes"]}])

    assert len(rows) == 36
    assert sum(row["label"] for row in rows) == 1
    assert any("Attribute: food.likes" in row["text"] for row in rows)
