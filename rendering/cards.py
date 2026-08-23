import html
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from request_orchestrator.models.evidence import EvidenceView

# messy but fine for now
def render_cards(
    items: Iterable[dict[str, Any]],
    per_row: int = 3,
    heading_key: str = "name",
    description_key: str = "description",
    image_key: str = "image_url",
    link_key: str = "url",
) -> None:
    items_list = list(items)
    per_row = max(1, per_row)
    image_base_dir = Path(os.getenv("PRODUCT_IMAGE_DIR", "db/images"))

    for start in range(0, len(items_list), per_row):
        row = items_list[start : start + per_row]
        cols = st.columns(per_row)
        for col, item in zip(cols, row):
            with col:
                with st.container(border=True):
                    image_url = item.get(image_key)
                    if isinstance(image_url, str) and image_url.strip():
                        image_value = image_url.strip()
                        if image_value.startswith(("http://", "https://")):
                            st.markdown(
                                (
                                    f'<img src="{html.escape(image_value, quote=True)}" '
                                    'style="width: 100%; height: auto;" />'
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            image_path = Path(image_value)
                            if image_path.is_absolute():
                                candidate = image_path
                            elif image_path.parts[:2] == ("db", "images"):
                                candidate = image_path
                            else:
                                candidate = image_base_dir / image_path
                            if candidate.exists():
                                st.image(str(candidate), width="stretch")

                    heading = item.get(heading_key) or "Untitled"
                    link = item.get(link_key)
                    if isinstance(link, str) and link.strip():
                        st.markdown(f"**[{heading}]({link})**")
                    else:
                        st.markdown(f"**{heading}**")

                    description = item.get(description_key)
                    if isinstance(description, str) and description.strip():
                        st.write(description)

                    price = item.get("price")
                    if price is not None:
                        st.caption(f"Price: {price}")


def render_magic_card_evidence_cards(items: Iterable[EvidenceView]) -> None:
    items_list = list(items)
    for start in range(0, len(items_list), 2):
        row = items_list[start : start + 2]
        cols = st.columns(2)
        for col, item in zip(cols, row):
            with col:
                _render_magic_card_evidence_card(item)


def render_magic_card_rulings(items: Iterable[EvidenceView]) -> None:
    rows = [
        {
            "Published": item.published_at.strip(),
            "Rule": item.summary.strip(),
        }
        for item in items
        if item.summary.strip()
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_magic_card_evidence_card(item: EvidenceView) -> None:
    with st.container(border=True):
        image_url = item.image_url.strip()
        if image_url:
            image_col, content_col = st.columns([1, 2])
        else:
            image_col, content_col = None, st.container()

        if image_col is not None:
            with image_col:
                st.image(image_url, width="stretch")

        with content_col:
            title = item.title.strip() or "Magic Card"
            card_url = item.url.strip()
            if card_url:
                st.markdown(f"**[{title}]({card_url})**")
            else:
                st.markdown(f"**{title}**")

            cmc = _raw_value(item.raw_payload, "cmc")
            colors = _string_list(_metadata_value(item.llm_metadata, "color_identity"))
            type_line = str(_metadata_value(item.llm_metadata, "type_line") or "").strip()
            color_text = ", ".join(colors) if colors else "Colorless"
            cmc_text = "Unknown" if cmc in (None, "") else str(cmc)
            st.caption(f"CMC {cmc_text} | {color_text}")
            if type_line:
                st.caption(type_line)

            oracle_text = str(_raw_value(item.raw_payload, "oracle_text") or "").strip()
            if oracle_text:
                st.write(oracle_text)

            pricing_rows = _pricing_rows(item.llm_metadata)
            if pricing_rows:
                st.dataframe(
                    pd.DataFrame(pricing_rows).reset_index(drop=True),
                    hide_index=True,
                    use_container_width=True,
                )

            if card_url:
                st.markdown(
                    (
                        "<a "
                        f'href="{html.escape(card_url, quote=True)}" '
                        'target="_blank" rel="noopener noreferrer" '
                        "style=\"display:inline-block;vertical-align:middle;margin-top:0.4rem;"
                        "padding:0.14rem 0.5rem;border:1px solid rgba(49,51,63,0.2);"
                        "border-radius:999px;text-decoration:none;font-size:0.82rem;"
                        "line-height:1.4;color:inherit;background:rgba(255,255,255,0.65);\">"
                        "↗ Open on Scryfall</a>"
                    ),
                    unsafe_allow_html=True,
                )


def _raw_value(raw_payload: Any, key: str) -> Any:
    if isinstance(raw_payload, dict):
        return raw_payload.get(key)
    return getattr(raw_payload, key, None)


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    return metadata.get(key)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _pricing_rows(source: Any) -> list[dict[str, str]]:
    pricing = _raw_value(source, "pricing")
    if not isinstance(pricing, list):
        return []

    rows: list[dict[str, str]] = []
    for entry in pricing:
        row = {
            "Set": str(_raw_value(entry, "set_name") or _raw_value(entry, "set") or "").strip(),
            "USD": str(_raw_value(entry, "usd") or "").strip(),
            "USD Foil": str(_raw_value(entry, "usd_foil") or "").strip(),
            "USD Etched": str(_raw_value(entry, "usd_etched") or "").strip(),
            "EUR": str(_raw_value(entry, "eur") or "").strip(),
            "EUR Foil": str(_raw_value(entry, "eur_foil") or "").strip(),
            "MTGO": str(_raw_value(entry, "tix") or _raw_value(entry, "magic_online") or "").strip(),
        }
        if any(value for value in row.values()):
            rows.append({label: value for label, value in row.items() if value})
    return rows
