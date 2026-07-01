import time
from pathlib import Path

import psycopg

from db.constants import DB_URL, MIGRATIONS_DIR, MIGRATION_TABLE


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


def ensure_migration_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def get_applied_versions(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT version FROM {MIGRATION_TABLE}")
        return {row[0] for row in cur.fetchall()}


def apply_migration(conn: psycopg.Connection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(
            f"INSERT INTO {MIGRATION_TABLE} (version) VALUES (%s)",
            (path.name,),
        )


def run_migrations() -> None:
    if not MIGRATIONS_DIR.exists():
        print("No migrations directory found; nothing to apply.")
        return

    wait_for_db()
    migration_paths = sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.is_file())

    with psycopg.connect(DB_URL) as conn:
        ensure_migration_table(conn)
        conn.commit()

        applied_versions = get_applied_versions(conn)
        pending = [path for path in migration_paths if path.name not in applied_versions]

        if not pending:
            print("No pending migrations.")
            return

        for path in pending:
            apply_migration(conn, path)
            conn.commit()
            print(f"Applied migration {path.name}")

    print("Database migrations complete.")


if __name__ == "__main__":
    run_migrations()
