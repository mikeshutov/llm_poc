from dataclasses import dataclass
from typing import Any, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from llm.clients.tool_response_parser import parse_tool_args

@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]

@dataclass(frozen=True)
class ToolCallResult:
    tool_calls: list[ToolCall]
    tool_calls_by_name: dict[str, list[ToolCall]]  # keeps all calls per name
    raw_message: Any

# We can probably expand on this client to be able to handle a bunch of different models not just openai models
class LlmClient:
    def __init__(self, client: Optional[OpenAI] = None, default_model: str = "gpt-4.1-mini"):
        self.client = client or OpenAI()
        self.default_model = default_model

    def call_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: Sequence[dict],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> ToolCallResult:
        resp = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            tools=list(tools),
            temperature=temperature,
        )

        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        tool_calls: list[ToolCall] = []
        tool_calls_by_name: dict[str, list[ToolCall]] = {}
        for tc in tcs:
            name = tc.function.name
            args = parse_tool_args(tc.function.arguments)

            call = ToolCall(name=name, args=args)
            tool_calls.append(call)
            tool_calls_by_name.setdefault(name, []).append(call)

        return ToolCallResult(
            tool_calls=tool_calls,
            tool_calls_by_name=tool_calls_by_name,
            raw_message=msg,
        )