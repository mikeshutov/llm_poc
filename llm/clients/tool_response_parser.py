import json
from typing import Any, Optional

def parse_tool_args(raw_args: Any) -> Optional[dict]:
    if raw_args is None:
        return None

    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        try:
            val = json.loads(raw_args)
            return val if isinstance(val, dict) else None
        except Exception:
            return None

    return None