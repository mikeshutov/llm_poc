from typing import Sequence

from common.message_constants import CONTENT_KEY, ROLE_ASSISTANT, ROLE_KEY, ROLE_USER


def compose_messages_from_roundtrips(
    roundtrips: Sequence,
):
    messages = []
    for rt in roundtrips:
        messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: rt.user_prompt})
        messages.append({ROLE_KEY: ROLE_ASSISTANT, CONTENT_KEY: rt.generated_response})
    return messages
