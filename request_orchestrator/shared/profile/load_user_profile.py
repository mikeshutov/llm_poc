from __future__ import annotations
from dataclasses import asdict

from langsmith import traceable

from personalization.profile.service import hydrate_user_profile_core, load_user_profile_attributes
from common.logging import create_conversation_event
from request_orchestrator.models.main_state import MainState
from rendering.debug import PROFILE_LOAD_KIND

ORCHESTRATOR_AGENT_NAME = "request_orchestrator"


@traceable(name="Load User Profile Node")
def load_user_profile(main_state: MainState) -> MainState:
    requested_attribute_types = list(main_state.request_analysis.requested_user_attribute_types)
    user_profile = main_state.execution_context.user_profile
    hydrate_user_profile_core(user_profile)
    load_user_profile_attributes(
        user_profile,
        requested_attribute_types=requested_attribute_types,
    )

    loaded_attributes = user_profile.user_attributes.attributes
    create_conversation_event(
        conversation_id=main_state.execution_context.conversation_id,
        roundtrip_id=main_state.execution_context.roundtrip_id,
        event_type=PROFILE_LOAD_KIND,
        source=ORCHESTRATOR_AGENT_NAME,
        agent_name=ORCHESTRATOR_AGENT_NAME,
        payload={
            "agent_name": ORCHESTRATOR_AGENT_NAME,
            "kind": PROFILE_LOAD_KIND,
            "data": {
                "first_name": user_profile.first_name,
                "last_name": user_profile.last_name,
                "display_name": user_profile.display_name,
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
        },
    )
    return main_state
