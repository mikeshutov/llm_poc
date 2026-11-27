from unittest.mock import MagicMock

import pytest


@pytest.fixture
def database_connection_factory():
    def create(*, rows=(), row=None):
        cursor = MagicMock()
        cursor.fetchall.return_value = list(rows)
        cursor.fetchone.return_value = row
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        return connection, cursor

    return create
