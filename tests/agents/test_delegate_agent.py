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
