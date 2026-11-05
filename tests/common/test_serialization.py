from datetime import datetime, timezone

from common.data import bounded_json_value, sanitize_for_json_storage


class FakeNumpyArray:
    __module__ = "numpy"

    def tolist(self):
        return [[1.0, 2.0], [3.0, 4.0]]


class FakeNumpyScalar:
    __module__ = "numpy"

    def item(self):
        return 1.5


def test_bounded_json_value_converts_datetime_to_json_safe_string() -> None:
    value = datetime(2026, 10, 31, 12, 34, 56, tzinfo=timezone.utc)

    result = bounded_json_value({"created_at": value})

    assert result == {"created_at": "2026-10-31T12:34:56+00:00"}


def test_bounded_json_value_removes_nul_characters_for_postgres_jsonb() -> None:
    assert bounded_json_value({"text": "before\x00after"}) == {"text": "beforeafter"}


def test_bounded_json_value_omits_binary_values() -> None:
    assert bounded_json_value({"payload": b"abc\x00def"}) == {
        "payload": "[binary data omitted: 7 bytes]"
    }


def test_json_serialization_normalizes_numpy_arrays_and_scalars() -> None:
    value = {"array": FakeNumpyArray(), "scalar": FakeNumpyScalar()}

    assert sanitize_for_json_storage(value) == {
        "array": [[1.0, 2.0], [3.0, 4.0]],
        "scalar": 1.5,
    }
    assert bounded_json_value(value) == {
        "array": [[1.0, 2.0], [3.0, 4.0]],
        "scalar": 1.5,
    }
