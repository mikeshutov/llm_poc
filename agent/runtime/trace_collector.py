from collections.abc import Callable

from agent.models.tool_call_trace import ToolCallTrace


class TraceCollector(list[ToolCallTrace]):
    def __init__(self, on_trace: Callable[[ToolCallTrace], None] | None = None):
        super().__init__()
        self._on_trace = on_trace

    def append(self, item: ToolCallTrace) -> None:
        super().append(item)
        if self._on_trace is not None:
            self._on_trace(item)
