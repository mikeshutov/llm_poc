import sys
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.shared.synthesis.prompts.synthesis_schema_prompt import SYNTHESIS_SCHEMA
from request_orchestrator.shared.synthesis.prompts.synthesis_rules import BASE_RULES


def test_synthesis_schema_forbids_displaying_evidence_ids_in_content() -> None:
    assert "never expose internal IDs in content" in SYNTHESIS_SCHEMA
    assert "including its agent prefix" in SYNTHESIS_SCHEMA
    assert '"main_agent:P1E1R1"' in SYNTHESIS_SCHEMA


def test_synthesis_rules_do_not_include_legacy_file_path_rendering_rule() -> None:
    assert not any("file_path" in rule for rule in BASE_RULES)
