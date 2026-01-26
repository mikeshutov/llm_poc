from typing import Sequence

# function used to convert roundtrips and prompt into a single payload
def compose_messages_from_roundtrips(
    roundtrips: Sequence,
):
    # convert all roundtrips into messages
    messages = []
    for rt in roundtrips:
        messages.append({"role": "user", "content": rt.user_prompt})
        messages.append({"role": "assistant", "content": rt.generated_response})
    return messages