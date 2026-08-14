from __future__ import annotations
from dataclasses import asdict

from langsmith import traceable

from personalization.profile.service import hydrate_user_profile_core, load_user_profile_attributes
from request_orchestrator.models.main_state import MainState
from rendering.debug import PROFILE_LOAD_KIND

ORCHESTRATOR_AGENT_NAME = "request_orchestrator"


@traceable(name="Load User Profile Node")
def load_user_profile(main_state: MainState) -> MainState:
    requested_attribute_types = list(main_state.request_analysis.requested_user_attribute_types)
    hydrate_user_profile_core(main_state.user_profile)
    load_user_profile_attributes(
        main_state.user_profile,
        requested_attribute_types=requested_attribute_types,
    )

    loaded_attributes = main_state.user_profile.user_attributes.attributes
    main_state.agent_log.add(
        agent_name=ORCHESTRATOR_AGENT_NAME,
        kind=PROFILE_LOAD_KIND,
        data={
            "first_name": main_state.user_profile.first_name,
            "last_name": main_state.user_profile.last_name,
            "display_name": main_state.user_profile.display_name,
            "requested_user_attribute_types": requested_attribute_types,
            "loaded_attribute_count": len(loaded_attributes),
            "loaded_attribute_types": sorted({attribute.attribute_type for attribute in loaded_attributes}),
            "loaded_attributes": [
                {
                    "attribute_type": attribute.attribute_type,
                    "group_key": attribute.group_key,
                    "value": list(attribute.value),
                }
                for attribute in loaded_attributes
            ],
        },
    )
    return main_state
