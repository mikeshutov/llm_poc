from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


class AttributeClassifierUnavailableError(RuntimeError):
    """Raised when the classifier service cannot be reached or returns an invalid response."""


@dataclass(frozen=True)
class AttributeClassifierClient:
    base_url: str = os.getenv("ATTRIBUTE_CLASSIFIER_SERVICE_URL", "http://localhost:5434")
    timeout_seconds: float = float(os.getenv("ATTRIBUTE_CLASSIFIER_TIMEOUT_SECONDS", "15"))

    def relevant_attributes(self, context: str, attributes: list[Any]) -> list[Any]:
        logger.info(
            "Sending attribute classification request: context=%r attributes=%r",
            context,
            attributes,
        )
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/classify",
                json={"context": context, "attributes": attributes},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            relevant = payload["relevant_attributes"]
            if not isinstance(relevant, list):
                raise ValueError("relevant_attributes must be a list")
            logger.info(
                "Received attribute classification result: relevant_attributes=%r",
                relevant,
            )
            return relevant
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise AttributeClassifierUnavailableError(
                "The local attribute classifier service is unavailable"
            ) from exc
