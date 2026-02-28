import os
import time
from pathlib import Path

import psycopg


DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/products")
SQL_FILES = [
    "db/extensions.sql",
    "db/product_schema.sql",
    "db/conversation_schema.sql",
    "db/conversation_roundtrip_schema.sql",
    "db/conversation_summary_schema.sql",
    "db/tool_calls_schema.sql",
]


def wait_for_db(timeout_s: int = 30) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with psycopg.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Database not ready after {timeout_s} seconds.")


def main() -> None:
    wait_for_db()
    for rel_path in SQL_FILES:
        path = Path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing SQL file: {rel_path}")
        sql = path.read_text(encoding="utf-8")
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
        print(f"Applied {rel_path}")

    print("Database setup complete.")


if __name__ == "__main__":
    main()
