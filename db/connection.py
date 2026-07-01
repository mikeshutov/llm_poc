from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from db.constants import DB_URL


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    conn.autocommit = True
    return conn
