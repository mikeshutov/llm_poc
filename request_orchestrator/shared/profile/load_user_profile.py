from __future__ import annotations

from dataclasses import asdict

from langsmith import traceable

from personalization.profile.service import hydrate_user_profile_core, load_user_profile_attributes
from request_orchestrator.models.agent_state import AgentState
from rendering.debug import PROFILE_LOAD_KIND


@traceable(name="Load User Profile Node")
def load_user_profile(agent_state: AgentState) -> AgentState:
    requested_attribute_types = list(agent_state.request_analysis.requested_user_attribute_types)
    hydrate_user_profile_core(agent_state.user_profile)
    load_user_profile_attributes(
        agent_state.user_profile,
        requested_attribute_types=requested_attribute_types,
    )

    loaded_attributes = agent_state.user_profile.user_attributes.attributes
    agent_state.log_status(
        agent_name=agent_state.agent_profile.name,
        kind=PROFILE_LOAD_KIND,
        data={
            "first_name": agent_state.user_profile.first_name,
            "last_name": agent_state.user_profile.last_name,
            "display_name": agent_state.user_profile.display_name,
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
    return agent_state
