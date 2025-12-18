import os
import random
from typing import Any

import psycopg
from datasets import load_dataset
from openai import OpenAI

DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/products")

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

def main():
    client = OpenAI()

    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    rows = [ds[i] for i in range(len(ds))]

    # Create stable IDs
    def row_id(i: int, r: dict[str, Any]) -> str:
        return str(r.get("id") or i)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM products;")
            existing = cur.fetchone()[0]
            if existing and existing > 0:
                print(f"products already has {existing} rows; exiting (delete table if you want reseed).")
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
                meta_batch.append((pid, name, category, color, style, gender, season, year, price))

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
                        (id, name, category, color, style, gender, season, year, price, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        [(*meta_batch[j], vectors[j]) for j in range(len(meta_batch))],
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
                    (id, name, category, color, style, gender, season, year, price, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [(*meta_batch[j], vectors[j]) for j in range(len(meta_batch))],
                )
                conn.commit()

    print("Done.")

if __name__ == "__main__":
    main()
