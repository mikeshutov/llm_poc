from agent_runtime.strategies.react.subagent_prompt import build_subagent_decision_prompt
from agents.delegate.profile import DELEGATE_PROFILE
from capabilities.registry.tools import TOOL_CATEGORIES


def test_delegate_domain_permissions_come_from_its_profile():
    assert DELEGATE_PROFILE.allowed_categories
    assert not DELEGATE_PROFILE.excluded_categories
    assert DELEGATE_PROFILE.allowed_categories <= TOOL_CATEGORIES.keys()
    assert {"web_fetch", "math_calculate", "discover_capabilities"} <= DELEGATE_PROFILE.tool_names
    assert "file_delete" not in DELEGATE_PROFILE.tool_names
    assert "writing_templates" not in DELEGATE_PROFILE.tool_names
    assert not DELEGATE_PROFILE.can_delegate_agents


def test_delegate_prompt_keeps_scope_without_prescribing_a_handoff(agent_state_factory):
    state = agent_state_factory(profile=DELEGATE_PROFILE, task="Check the sources")

    prompt, _ = build_subagent_decision_prompt(state)

    assert DELEGATE_PROFILE.planner_instruction in prompt.instruction
    assert "Stay within its scope" in prompt.instruction
    assert "supporting evidence" in prompt.instruction
    assert "Return the completed result" not in prompt.instruction
    assert "parent's final report" not in prompt.instruction
