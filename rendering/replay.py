from __future__ import annotations

import streamlit as st

REPLAY_TARGET_KEY = "replay_target"


def clear_replay_state() -> None:
    st.session_state.pop(REPLAY_TARGET_KEY, None)


def request_replay(roundtrip_id: str) -> None:
    st.session_state[REPLAY_TARGET_KEY] = {"roundtrip_id": roundtrip_id}


def pop_replay_target() -> dict[str, str] | None:
    target = st.session_state.pop(REPLAY_TARGET_KEY, None)
    return target if isinstance(target, dict) else None
