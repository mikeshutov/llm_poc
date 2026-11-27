from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor


def test_open_meteo_package_imports_concurrently_without_partial_module_failure() -> None:
    module_names = [
        "integrations.open_meteo",
        "integrations.open_meteo.client",
        "integrations.open_meteo.errors",
        "integrations.open_meteo.models",
    ]
    with ThreadPoolExecutor(max_workers=len(module_names)) as executor:
        modules = list(executor.map(importlib.import_module, module_names))

    assert all(module is not None for module in modules)
    assert hasattr(modules[0], "OpenMeteoClient")
