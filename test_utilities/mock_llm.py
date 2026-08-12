from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Sequence


@dataclass(frozen=True)
class MockLLMResponse:
    content: str


@dataclass
class MockLLMScenario:
    request_analysis: str | None = None
    profile_planner: list[str] = field(default_factory=list)
    main_planner: list[str] = field(default_factory=list)
    synthesis: list[str] = field(default_factory=list)
    evaluator: list[str] = field(default_factory=list)
    fallback: list[str] = field(default_factory=list)


@dataclass
class MockLLM:
    responses: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def __init__(self, responses: Sequence[str] | MockLLMScenario | None = None):
        self.prompts = []
        self.invocations = []
        self._lock = Lock()
        self._scenario = responses if isinstance(responses, MockLLMScenario) else None
        self.responses = [] if self._scenario is not None else list(responses or [])

    def push_response(self, content: str) -> None:
        with self._lock:
            self.responses.append(content)

    def _is_request_analysis_prompt(self, prompt: str) -> bool:
        return 'Response Schema:' in prompt and 'requested_user_attribute_types' in prompt

    def _is_synthesis_prompt(self, prompt: str) -> bool:
        return (
            'Evidence (JSON):' in prompt
            or '  "next_question": ' in prompt
            or prompt.startswith('Solve the following task or problem.')
        )

    def _is_evaluator_prompt(self, prompt: str) -> bool:
        return prompt.startswith('You are an evaluator between planning and synthesis.')

    def _is_profile_planner_prompt(self, prompt: str) -> bool:
        return (
            'Review this turn for durable user attribute maintenance needs.' in prompt
            or 'Your scope is limited maintaining the user attributes and profile.' in prompt
            or 'Maintain durable user profile attributes.' in prompt
            or 'Profile policy:' in prompt
        )

    def _pop_from_scenario(self, prompt: str) -> str:
        assert self._scenario is not None

        if self._is_request_analysis_prompt(prompt):
            if self._scenario.request_analysis is None:
                raise AssertionError('MockLLM request-analysis response was not configured.')
            response = self._scenario.request_analysis
            self._scenario.request_analysis = None
            return response

        if self._is_evaluator_prompt(prompt):
            if self._scenario.evaluator:
                return self._scenario.evaluator.pop(0)
            return '{"status": "SATISFIED", "relevant_evidence": [], "missing_information": [], "refined_goal": ""}'

        if self._is_synthesis_prompt(prompt):
            if self._scenario.synthesis:
                return self._scenario.synthesis.pop(0)
            raise AssertionError('MockLLM synthesis response queue is empty.')

        if self._is_profile_planner_prompt(prompt):
            if self._scenario.profile_planner:
                return self._scenario.profile_planner.pop(0)
            raise AssertionError('MockLLM profile-planner response queue is empty.')

        if self._scenario.main_planner:
            return self._scenario.main_planner.pop(0)

        if self._scenario.fallback:
            return self._scenario.fallback.pop(0)

        raise AssertionError('MockLLM main-planner response queue is empty.')

    def invoke(self, prompt: str, *args: Any, **kwargs: Any) -> MockLLMResponse:
        with self._lock:
            self.prompts.append(prompt)
            self.invocations.append(
                {
                    'prompt': prompt,
                    'args': args,
                    'kwargs': kwargs,
                }
            )

            if self._scenario is not None:
                return MockLLMResponse(content=self._pop_from_scenario(prompt))

            if not self.responses:
                raise AssertionError('MockLLM.invoke was called without a queued response.')

            return MockLLMResponse(content=self.responses.pop(0))

    @property
    def last_prompt(self) -> str | None:
        return self.prompts[-1] if self.prompts else None
