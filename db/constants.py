from __future__ import annotations

import os
from pathlib import Path

DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/products")
MIGRATIONS_DIR = Path("db/migrations")
MIGRATION_TABLE = "schema_migrations"
