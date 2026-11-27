import pytest

from common.cache.repository import decode_cache_output, encode_cache_output


class FakeTool:
    def __init__(self, name="test_tool", callback=None) -> None:
        self.name = name
        self.description = ""
        self.args_schema = None
        self._callback = (lambda args: {}) if callback is None else callback

    def invoke(self, tool_input=None):
        return self._callback(tool_input or {})


class FakeCacheRepository:
    def __init__(self):
        self.entries = {}
        self.get_calls = 0
        self.put_calls = 0

    @staticmethod
    def sanitize(value):
        return value

    @staticmethod
    def build_cache_key(*, tool_name, tool_version, tool_input, user_id=None):
        return (tool_name, tool_version, user_id, repr(tool_input))

    def get(self, *, tool, tool_input, user_id=None):
        self.get_calls += 1
        cache_key = self.build_cache_key(
            tool_name=tool.name,
            tool_version=tool.version,
            tool_input=tool_input,
            user_id=user_id,
        )
        cached_output = self.entries.get(cache_key)
        return None if cached_output is None else decode_cache_output(cached_output)

    def put(self, **kwargs):
        self.put_calls += 1
        cache_key = self.build_cache_key(
            tool_name=kwargs["tool"].name,
            tool_version=kwargs["tool"].version,
            tool_input=kwargs["tool_input"],
            user_id=kwargs.get("user_id"),
        )
        self.entries[cache_key] = encode_cache_output(kwargs["output_payload"])


class FailingCacheRepository(FakeCacheRepository):
    def __init__(self, *, fail_get=False, fail_put=False):
        super().__init__()
        self.fail_get = fail_get
        self.fail_put = fail_put

    def get(self, **kwargs):
        self.get_calls += 1
        if self.fail_get:
            raise RuntimeError("cache lookup unavailable")
        return super().get(**kwargs)

    def put(self, **kwargs):
        self.put_calls += 1
        if self.fail_put:
            raise RuntimeError("cache store unavailable")
        super().put(**kwargs)


@pytest.fixture
def tool_factory():
    return FakeTool


@pytest.fixture
def cache_repository_factory():
    def create(*, fail_get=False, fail_put=False):
        if fail_get or fail_put:
            return FailingCacheRepository(fail_get=fail_get, fail_put=fail_put)
        return FakeCacheRepository()

    return create
