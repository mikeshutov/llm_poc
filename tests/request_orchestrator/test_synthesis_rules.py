from request_orchestrator.shared.synthesis.prompts.synthesis_rules import (
    PROFILE_PERSONALIZATION_RULE,
    TONE_ADAPTATION_RULE,
    build_synthesis_rules,
)


def test_tone_adaptation_rule_is_conditional() -> None:
    assert TONE_ADAPTATION_RULE not in build_synthesis_rules()
    assert TONE_ADAPTATION_RULE in build_synthesis_rules(apply_tone_preferences=True)


def test_profile_personalization_rule_is_conditional() -> None:
    assert PROFILE_PERSONALIZATION_RULE not in build_synthesis_rules()
    assert PROFILE_PERSONALIZATION_RULE in build_synthesis_rules(
        apply_profile_personalization=True,
    )
