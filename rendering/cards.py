import os
from pathlib import Path
from typing import Any, Iterable

import streamlit as st

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
                            st.image(image_value, width="stretch")
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
