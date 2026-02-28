from enum import Enum


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
