from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage


@pytest.fixture
def tool_exchange_factory():
    def create(*, tool="web_search", call_id=None, args=None, content="Results", call_overrides=None, result_overrides=None):
        call_id = str(uuid4()) if call_id is None else call_id
        call = AIMessage(**{
            "content": "", "tool_calls": [{"id": call_id, "name": tool, "args": deepcopy(args or {})}],
            **deepcopy(call_overrides or {}),
        })
        result = ToolMessage(**{
            "content": content, "tool_call_id": call.tool_calls[0]["id"], "name": call.tool_calls[0]["name"],
            **deepcopy(result_overrides or {}),
        })
        return call, result

    return create


class FakeLangChainResponse:
    def __init__(
        self, content="{}", model_name="gpt-5.6-luna", tool_calls=None, *,
        input_tokens=100, output_tokens=20,
    ):
        self.content = content
        self.tool_calls = deepcopy(tool_calls or [])
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        self.response_metadata = {"model_name": model_name}


class FakeInvokeLLM:
    def __init__(self, content="{}", model_name="gpt-5.6-luna", tool_calls=None, **usage):
        self.content = content
        self.model_name = model_name
        self.tool_calls = deepcopy(tool_calls or [])
        self.usage = usage
        self.prompts = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeLangChainResponse(self.content, self.model_name, self.tool_calls, **self.usage)


class FakeOpenAIClient:
    def __init__(self, *, responses=()):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.responses = list(responses)

    def create(self, **kwargs):
        return self.responses.pop(0)


@pytest.fixture
def langchain_response_factory():
    return FakeLangChainResponse


@pytest.fixture
def llm_factory():
    return FakeInvokeLLM


@pytest.fixture
def openai_client_factory():
    return FakeOpenAIClient


@pytest.fixture
def embedding_client_factory():
    def create(*, embedding=(1.0, 0.0), error=None):
        request = Mock(
            return_value=SimpleNamespace(data=[SimpleNamespace(embedding=list(embedding))]),
            side_effect=error,
        )
        return SimpleNamespace(embeddings=SimpleNamespace(create=request))

    return create


@pytest.fixture
def openai_response_factory():
    def create(*, input_tokens=210, output_tokens=45):
        return SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=input_tokens, completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ))

    return create


class TrackingChatOpenAI:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def invoke(self, prompt: str):
        raise AssertionError(f'Unexpected invoke for prompt: {prompt[:80]}')


class TrackingChatAnthropic:
    def __init__(self, model_name: str, **kwargs):
        self.model = model_name
        self.kwargs = kwargs

    def invoke(self, prompt: str):
        raise AssertionError(f'Unexpected invoke for prompt: {prompt[:80]}')

@pytest.fixture
def tracking_openai_model_factory():
    return TrackingChatOpenAI


@pytest.fixture
def tracking_anthropic_model_factory():
    return TrackingChatAnthropic
