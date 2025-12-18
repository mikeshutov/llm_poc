# intent_processing/category_resolution.py
from functools import lru_cache
import sqlite3
from pathlib import Path
from difflib import get_close_matches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "products.db"
TABLE_NAME = "products"


@lru_cache(maxsize=1)
def get_all_categories() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(f"SELECT DISTINCT category FROM {TABLE_NAME}").fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


def _normalize(s: str) -> str:
    return s.strip().lower()

# Hand-tuned aliases for common LLM outputs → your catalog categories
CATEGORY_ALIAS_MAP = {
    "shirt": "Shirts",
    "tshirt": "Tshirts",
    "t-shirt": "Tshirts",
    "t shirt": "Tshirts",
    "jeans": "Jeans",
    "pants": "Trousers",
    "trousers": "Trousers",
    "hoodie": "Sweatshirts",
    "jacket": "Jackets",
    "coat": "Jackets",
    "sneakers": "Shoes",
    "shoes": "Shoes",
    "heels": "Heels",
    "sandals": "Sandals",
    "bag": "Bags",
    "dress": "Dresses",
    "skirt": "Skirts",
}


def resolve_category(llm_category: str) -> str | None:
    """
    Map a free-text LLM category (e.g. 'shirt') to a catalog category
    (e.g. 'Shirts', 'Tshirts'). Returns None if we can't map confidently.
    """
    if not llm_category:
        return None

    all_cats = get_all_categories()
    if not all_cats:
        return None

    norm = _normalize(llm_category)

    # 1) Exact alias first
    if norm in CATEGORY_ALIAS_MAP:
        return CATEGORY_ALIAS_MAP[norm]

    # 2) Exact normalized match to existing categories
    normalized_map = { _normalize(c): c for c in all_cats }
    if norm in normalized_map:
        return normalized_map[norm]

    # 3) Fuzzy match as a fallback (difflib)
    candidates = get_close_matches(norm, normalized_map.keys(), n=1, cutoff=0.7)
    if candidates:
        best_norm = candidates[0]
        return normalized_map[best_norm]

    # 4) Give up – treat as no category filter
    return None