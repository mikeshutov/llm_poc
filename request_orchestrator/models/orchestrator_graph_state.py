from __future__ import annotations

from dataclasses import dataclass
from request_orchestrator.models.main_state import MainState


@dataclass
class OrchestratorGraphState:
    main_state: MainState
