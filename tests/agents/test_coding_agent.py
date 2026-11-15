from agents.coding.profile import CODING_PROFILE, build_coding_profile
from agents.seeder import SYSTEM_AGENT_PROFILES
from agent_runtime.models.agent_profile import AgentExecutionStrategy, AgentName


def test_coding_agent_is_general_purpose_and_seeded() -> None:
    assert CODING_PROFILE.name == AgentName.CODING
    assert CODING_PROFILE.description == (
        "An agent capable of generating, executing, and validating code to solve computational tasks."
    )
    assert CODING_PROFILE.execution_strategy == AgentExecutionStrategy.REACT
    assert CODING_PROFILE in SYSTEM_AGENT_PROFILES


def test_coding_profile_can_be_rebuilt() -> None:
    profile = build_coding_profile()

    assert {tool.name for tool in profile.tools} == {
        "math_calculate",
        "code_execute",
        "finance_exchange_rates",
        "finance_stock_price",
        "finance_crypto_markets",
        "finance_crypto_historical_prices",
        "memory_lookup_evidence",
        "web_fetch",
        "workspace_file_write",
        "workspace_file_read",
        "workspace_file_edit",
        "workspace_file_list",
        "workspace_file_delete",
        "workspace_file_promote",
    }
