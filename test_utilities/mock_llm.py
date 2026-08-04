from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class MockLLMResponse:
    content: str


@dataclass
class MockLLM:
    responses: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def __init__(self, responses: Sequence[str] | None = None):
        self.responses = list(responses or [])
        self.prompts = []
        self.invocations = []

    def push_response(self, content: str) -> None:
        self.responses.append(content)

    def invoke(self, prompt: str, *args: Any, **kwargs: Any) -> MockLLMResponse:
        self.prompts.append(prompt)
        self.invocations.append(
            {
                "prompt": prompt,
                "args": args,
                "kwargs": kwargs,
            }
        )

        if not self.responses:
            raise AssertionError("MockLLM.invoke was called without a queued response.")

        return MockLLMResponse(content=self.responses.pop(0))

    @property
    def last_prompt(self) -> str | None:
        return self.prompts[-1] if self.prompts else None
