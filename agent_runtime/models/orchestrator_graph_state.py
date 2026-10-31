from __future__ import annotations

from dataclasses import dataclass
from agent_runtime.models.main_state import MainState


@dataclass
class OrchestratorGraphState:
    main_state: MainState
