from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_VALUES


def load_source_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"line {line_number}: text must be a non-empty string")
        labels = record.get("labels")
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"line {line_number}: labels must be a list of strings")
        unknown_labels = set(labels) - set(ATTRIBUTE_TYPE_VALUES)
        if unknown_labels:
            raise ValueError(f"line {line_number}: unknown labels: {sorted(unknown_labels)}")
        examples.append({"text": record["text"].strip(), "labels": sorted(set(labels))})
    return examples


def build_binary_examples(source_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in source_examples:
        positive_labels = set(example["labels"])
        for attribute in ATTRIBUTE_TYPE_VALUES:
            rows.append(
                {
                    "text": f"Context: {example['text']}\nAttribute: {attribute}",
                    "label": int(attribute in positive_labels),
                }
            )
    return rows


def train(source_path: Path, output_path: Path, base_model: str) -> None:
    from datasets import Dataset
    from setfit import SetFitModel, SetFitTrainer

    source_examples = load_source_examples(source_path)
    if not source_examples:
        raise ValueError("training data is empty")

    trainer = SetFitTrainer(
        model=SetFitModel.from_pretrained(base_model),
        train_dataset=Dataset.from_list(build_binary_examples(source_examples)),
        column_mapping={"text": "text", "label": "label"},
        num_iterations=20,
        num_epochs=1,
        batch_size=16,
    )
    trainer.train()
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SetFit attribute relevance classifier")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("training_data.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "trained_models" / "attribute-relevance-setfit",
    )
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    train(args.data, args.output, args.base_model)
