from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retry_on_timeout: bool = True
    retry_on_429: bool = True
    retry_on_5xx: bool = True
    backoff_seconds: float = 1.0
