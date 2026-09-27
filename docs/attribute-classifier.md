# Attribute relevance classifier

`services/attribute_classifier` exposes a SetFit binary classifier that selects which candidate attributes matter for a piece of context.

The SetFit checkpoint must be a model trained with class `0 = irrelevant` and class `1 = relevant`. Configure it with `ATTRIBUTE_CLASSIFIER_MODEL`; the placeholder default is intentionally not a downloadable model.

Start it with after training a local checkpoint:

```text
docker compose up --build attribute-classifier
```

Compose mounts `services/attribute_classifier/training/trained_models` into the service container and loads `/app/services/attribute_classifier/training/trained_models/attribute-relevance-setfit` by default. Set `ATTRIBUTE_CLASSIFIER_MODEL` explicitly if you want to load a different local path or a Hugging Face model.

Request:

```json
{
  "context": "Find a waterproof jacket for winter hiking",
  "attributes": [
    {"id": "waterproof", "text": "waterproof rating"},
    {"id": "color", "text": "preferred color"},
    "insulation type"
  ]
}
```

`POST /classify` returns `relevant_attributes` in the same format as the input, plus `results` containing each candidate's score and boolean decision. The threshold defaults to `0.5` and can be changed with `ATTRIBUTE_CLASSIFIER_THRESHOLD`.

The application-side `AttributeClassifierClient` is available in `attribute_classifier/client.py` and uses `ATTRIBUTE_CLASSIFIER_SERVICE_URL` (default `http://localhost:5434`).

## Basic training

Seed examples live in `services/attribute_classifier/training/training_data.jsonl` using this format:

```json
{"text":"Recommend some RPGs I'd probably like","labels":["media.likes"]}
```

Add more examples to that file, then install the training dependencies and run:

```text
pip install -r services/attribute_classifier/training/training-requirements.txt
python -m services.attribute_classifier.training.train
```

The script expands each source example across every available attribute type and trains a binary SetFit classifier (`0 = irrelevant`, `1 = relevant`). The trained checkpoint is written to `services/attribute_classifier/training/trained_models/attribute-relevance-setfit`. Start the service with `ATTRIBUTE_CLASSIFIER_MODEL=services/attribute_classifier/training/trained_models/attribute-relevance-setfit` after training.
