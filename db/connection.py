from __future__ import annotations

import os

import psycopg
from psycopg.rows import dict_row

DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/products")


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    conn.autocommit = True
    return conn
