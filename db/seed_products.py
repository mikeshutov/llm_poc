import os
import random
from pathlib import Path
from typing import Any

import psycopg
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/products")
IMAGE_DIR = Path(os.getenv("PRODUCT_IMAGE_DIR", "db/images"))

# arbitrary price range since the products do not have a price
CATEGORY_PRICE_RANGES = {
    "Tshirts": (10, 40),
    "Shirts": (15, 70),
    "Jeans": (30, 120),
    "Trousers": (20, 90),
    "Dresses": (25, 200),
    "Jackets": (50, 300),
    "Sweaters": (20, 150),
    "Shoes": (40, 200),
    "Heels": (30, 150),
    "Sandals": (15, 80),
    "Bags": (20, 120),
    "Accessories": (5, 50),
}

def pick_category(row: dict[str, Any]) -> str | None:
    return row.get("articleType") or row.get("subCategory") or row.get("masterCategory")

def gen_price(cat: str | None) -> float:
    low, high = CATEGORY_PRICE_RANGES.get(cat or "", (10, 100))
    return round(random.uniform(low, high), 2)

def build_embedding_text(row: dict[str, Any]) -> str:
    # We can start with simple name and usage embeddings
    # we may want other embeddings depending on the thing
    # we are searching for more contextual searches
    name = row.get("productDisplayName") or ""
    style = row.get("usage") or ""
    return f"Name: {name}\nStyle: {style}"

def extract_image_url(value: Any) -> str | None:
    def is_thumbnail(val: str) -> bool:
        lowered = val.lower()
        return "thumb" in lowered or "thumbnail" in lowered

    if isinstance(value, str):
        return None if is_thumbnail(value) else value
    if isinstance(value, dict):
        candidates: list[str] = []
        for key in ("url", "path", "image_url"):
            v = value.get(key)
            if isinstance(v, str) and v:
                candidates.append(v)
        for v in candidates:
            if not is_thumbnail(v):
                return v
        if candidates:
            return candidates[0]
        return None
    filename = getattr(value, "filename", None)
    if isinstance(filename, str) and filename:
        return filename
    return None

def save_image(value: Any, pid: str) -> str | None:
    if value is None:
        return None
    save_path = IMAGE_DIR / f"{pid}.jpg"
    try:
        value.save(save_path, format="JPEG")
    except Exception:
        pass
    if save_path.exists():
        return str(save_path)
    return None

def main():
    allow_image_backfill = os.getenv("ALLOW_IMAGE_BACKFILL", "0") == "1"
    force_image_refresh = os.getenv("FORCE_IMAGE_REFRESH", "0") == "1"
    client = OpenAI() if not allow_image_backfill else None
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    rows = [ds[i] for i in range(len(ds))]
    # Create stable IDs
    def row_id(i: int, r: dict[str, Any]) -> str:
        return str(r.get("id") or i)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM products;")
            existing = cur.fetchone()[0]
            if existing and existing > 0 and not allow_image_backfill:
                print(f"products already has {existing} rows; exiting (delete table if you want reseed).")
                return

        if allow_image_backfill:
            with conn.cursor() as cur:
                update_batch: list[tuple] = []
                for i, r in enumerate(rows):
                    pid = row_id(i, r)
                    img = r.get("image")
                    image_url = save_image(img, pid)
                    if image_url is None:
                        image_url = (
                            extract_image_url(r.get("image"))
                            or extract_image_url(r.get("img"))
                            or extract_image_url(r.get("image_url"))
                        )

                    if not image_url:
                        continue

                    update_batch.append((image_url, pid))

                    if len(update_batch) >= 500:
                        cur.executemany(
                            """
                            UPDATE products
                            SET image_url = %s
                            WHERE id = %s
                            """,
                            update_batch,
                        )
                        conn.commit()
                        update_batch.clear()
                        print(f"Backfilled {i+1}/{len(rows)}")

                if update_batch:
                    cur.executemany(
                        """
                        UPDATE products
                        SET image_url = %s
                        WHERE id = %s
                        """,
                        update_batch,
                    )
                    conn.commit()
            print("Done.")
            return

        BATCH = int(os.getenv("EMBED_BATCH_SIZE", "128"))
        embed_batch: list[str] = []
        meta_batch: list[tuple] = []

        with conn.cursor() as cur:
            for i, r in enumerate(rows):
                pid = row_id(i, r)
                name = str(r.get("productDisplayName") or "")
                category = pick_category(r)
                color = r.get("baseColour")
                style = r.get("usage")
                gender = r.get("gender")
                season = r.get("season")
                year = int(r.get("year") or 0)
                price = gen_price(category)

                embed_batch.append(build_embedding_text(r))
                img = r.get("image")
                image_url = save_image(img, pid)
                if image_url is None:
                    image_url = (
                        extract_image_url(r.get("image"))
                        or extract_image_url(r.get("img"))
                        or extract_image_url(r.get("image_url"))
                    )
                meta_batch.append((pid, name, category, color, style, gender, season, year, price, image_url))

                if len(embed_batch) >= BATCH:
                    # Embeddings call (batch). Use an embedding model (not chat).
                    emb = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=embed_batch,
                    )
                    vectors = [d.embedding for d in emb.data]

                    cur.executemany(
                        """
                        INSERT INTO products
                        (id, name, category, color, style, gender, season, year, price, image_url, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE
                        SET image_url = EXCLUDED.image_url
                        WHERE %s OR products.image_url IS NULL OR products.image_url = '';
                        """,
                        [(*meta_batch[j], force_image_refresh) for j in range(len(meta_batch))],
                    )
                    conn.commit()

                    embed_batch.clear()
                    meta_batch.clear()
                    print(f"Inserted {i+1}/{len(rows)}")

            # Flush remainder
            if embed_batch:
                emb = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=embed_batch,
                )
                vectors = [d.embedding for d in emb.data]
                cur.executemany(
                    """
                    INSERT INTO products
                    (id, name, category, color, style, gender, season, year, price, image_url, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE
                    SET image_url = EXCLUDED.image_url
                    WHERE %s OR products.image_url IS NULL OR products.image_url = '';
                    """,
                    [(*meta_batch[j], force_image_refresh) for j in range(len(meta_batch))],
                )
                conn.commit()

    print("Done.")

if __name__ == "__main__":
    main()
