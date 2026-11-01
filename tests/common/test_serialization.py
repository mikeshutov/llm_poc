from datetime import datetime, timezone

from common.data import bounded_json_value


def test_bounded_json_value_converts_datetime_to_json_safe_string() -> None:
    value = datetime(2026, 10, 31, 12, 34, 56, tzinfo=timezone.utc)

    result = bounded_json_value({"created_at": value})

    assert result == {"created_at": "2026-10-31T12:34:56+00:00"}
