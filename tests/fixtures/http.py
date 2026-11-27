import json

import pytest
from requests import Response


@pytest.fixture
def http_response_factory():
    def create(*, payload=None, text="", status_code=200, headers=None, url="https://example.com"):
        response = Response()
        response.status_code = status_code
        response.url = url
        response.encoding = "utf-8"
        response.headers.update(headers or {})
        response._content = (text if payload is None else json.dumps(payload)).encode("utf-8")
        return response

    return create
