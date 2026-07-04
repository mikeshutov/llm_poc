from __future__ import annotations

from uuid import UUID

import streamlit as st

REPLAY_TARGET_KEY = "replay_target"


def clear_replay_state() -> None:
    st.session_state.pop(REPLAY_TARGET_KEY, None)


def request_replay(roundtrip_id: str) -> None:
    st.session_state[REPLAY_TARGET_KEY] = {"roundtrip_id": roundtrip_id}


def pop_replay_target() -> dict[str, str] | None:
    target = st.session_state.pop(REPLAY_TARGET_KEY, None)
    return target if isinstance(target, dict) else None


def render_replay_control(*, roundtrip_id: UUID | str | None, timestamp: str | None = None) -> None:
    if roundtrip_id is None and not timestamp:
        return

    rid = str(roundtrip_id) if roundtrip_id is not None else None
    col_caption, col_replay = st.columns([11, 1], vertical_alignment="center")
    with col_caption:
        if timestamp:
            st.caption(timestamp)
    with col_replay:
        if rid is not None:
            st.button(
                ":material/replay:",
                key=f"replay_{rid}",
                help="Replay from this prompt",
                on_click=request_replay,
                args=(rid,),
            )
