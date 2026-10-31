# Shared node / prompt identifiers
from enum import StrEnum


class NodeName(StrEnum):
    """Canonical producer IDs used by orchestration nodes and node logs."""

    LOAD_AGENTS = "load_agents"
    ENRICH_STATE = "enrich_state"
    LOAD_USER_PROFILE = "load_user_profile"
    MAIN_REQUEST_STRATEGY = "main_request_strategy"
    DISCOVER_AGENT_STATE = "discover_agent_state"
    PLANNER = "planner"
    EXECUTOR = "execute_tools"
    EVALUATOR = "evaluator"
    VALIDATE_EXECUTION_RESULT = "validate_execution_result"
    SYNTHESIS = "synthesis"
    SYNTHESIZE = "synthesize"
    REASON = "reason"
    ATTRIBUTE_DISCOVERY = "attribute_discovery"
    CAPABILITY_DISCOVERY = "capability_discovery"
    COMPLETION_EVALUATOR = "completion_evaluator"
    RERANKER = "reranker"


# Graph edges
LOAD_AGENTS_EDGE = NodeName.LOAD_AGENTS.value
ENRICH_STATE_EDGE = NodeName.ENRICH_STATE.value
PROFILE_LOADING_EDGE = NodeName.LOAD_USER_PROFILE.value
SYNTHESIZE_EDGE = NodeName.SYNTHESIZE.value
PLAN_EDGE = NodeName.PLANNER.value
EXECUTE_TOOLS_EDGE = NodeName.EXECUTOR.value
EVALUATE_EDGE = NodeName.EVALUATOR.value

DISCOVER_AGENT_STATE_EDGE = NodeName.DISCOVER_AGENT_STATE.value
