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
    SYNTHESIS = "synthesize"
    INITIAL_DECISION = "initial_decision"
    REASON = "reason"
    ATTRIBUTE_DISCOVERY = "attribute_discovery"
    CAPABILITY_DISCOVERY = "capability_discovery"
    COMPLETION_EVALUATOR = "completion_evaluator"
    RERANKER = "reranker"


PLANNER_KIND = "planner"
EVALUATOR_KIND = "evaluator"
SYNTHESIS_KIND = "synthesis"
SYNTHESIZE_KIND = "synthesize"
EXECUTE_TOOLS_KIND = "execute_tools"

# Graph edges
LOAD_AGENTS_EDGE = NodeName.LOAD_AGENTS.value
ENRICH_STATE_EDGE = NodeName.ENRICH_STATE.value
PROFILE_LOADING_EDGE = NodeName.LOAD_USER_PROFILE.value
SYNTHESIZE_EDGE = NodeName.SYNTHESIS.value
PLAN_EDGE = NodeName.PLANNER.value
EXECUTE_TOOLS_EDGE = NodeName.EXECUTOR.value
EVALUATE_EDGE = NodeName.EVALUATOR.value

# Prompt kinds
PLANNER_PROMPT_KIND = PLANNER_KIND
EVALUATOR_PROMPT_KIND = EVALUATOR_KIND
SYNTHESIS_PROMPT_KIND = SYNTHESIS_KIND
DISCOVER_AGENT_STATE_EDGE = NodeName.DISCOVER_AGENT_STATE.value
