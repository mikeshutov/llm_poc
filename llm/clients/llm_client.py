import base64
from time import perf_counter
from typing import Any, Optional, Sequence

from openai import OpenAI

from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_SYSTEM, ROLE_USER
from conversation.models.conversation_model_config import ConversationModelConfig
from llm.clients.tool_response_parser import parse_tool_args
from llm.models.tool_call import ToolCall, ToolCallResult
from llm.usage import record_llm_call
from request_orchestrator.shared.runtime_context import get_current_conversation_id, get_current_roundtrip_id, get_current_user_id

CAPTION_MAX_TOKENS = 200

# mostly used for embeddings still
def get_openai_client() -> OpenAI:
    return OpenAI()

# used for the rest of our requests
def get_llm_client(default_model: str = ConversationModelConfig.default_main_agent_planner_model()) -> "LlmClient":
    return LlmClient(default_model=default_model)


# This is mostly for when we want to utilize our own LLM client
# We can probably expand on this client to be able to handle a bunch of different models not just openai models
class LlmClient:
    def __init__(self, client: Optional[OpenAI] = None, default_model: str = ConversationModelConfig.default_main_agent_planner_model()):
        self.client = client or get_openai_client()
        self.default_model = default_model

    def call_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: Sequence[dict],
        model: Optional[str] = None,
        temperature: float | None = None,
    ) -> ToolCallResult:
        resolved_model = model or self.default_model
        started_at = perf_counter()
        resp = self.client.chat.completions.create(
            model=resolved_model,
            messages=[
                {ROLE_KEY: ROLE_SYSTEM, CONTENT_KEY: system_prompt},
                *messages,
            ],
            tools=list(tools),
            **({"temperature": temperature} if temperature is not None else {}),
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        record_llm_call(
            raw_response=resp,
            model_name=resolved_model,
            conversation_id=get_current_conversation_id(),
            roundtrip_id=get_current_roundtrip_id(),
            user_id=get_current_user_id(),
            agent="utility",
            stage="tool_calling",
            callsite="llm_client.call_with_tools",
            metadata={"tool_count": len(tools)},
            latency_ms=latency_ms,
            input_object={
                "messages": [
                    {ROLE_KEY: ROLE_SYSTEM, CONTENT_KEY: system_prompt},
                    *messages,
                ],
                "tools": list(tools),
                "temperature": temperature,
            },
            output_object={
                "raw_message": getattr(resp.choices[0], "message", None),
            },
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

    def generate_caption_from_image_file(self, path: str) -> str:
        with open(path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        started_at = perf_counter()
        response = self.client.chat.completions.create(
            model=self.default_model,
            messages=[
                {
                    ROLE_KEY: ROLE_USER,
                    CONTENT_KEY: [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image for semantic search. "
                                "Include details such as objects, colors, text, and context. Be specific yet concise."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=CAPTION_MAX_TOKENS,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        record_llm_call(
            raw_response=response,
            model_name=self.default_model,
            conversation_id=get_current_conversation_id(),
            roundtrip_id=get_current_roundtrip_id(),
            user_id=get_current_user_id(),
            agent="utility",
            stage="image_caption",
            callsite="llm_client.generate_caption_from_image_file",
            metadata={"max_completion_tokens": CAPTION_MAX_TOKENS},
            latency_ms=latency_ms,
        )

        return response.choices[0].message.content
