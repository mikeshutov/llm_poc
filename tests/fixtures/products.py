from copy import deepcopy

import pytest

from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.models.product_source import ProductSource


@pytest.fixture
def product_factory():
    def create(**overrides):
        return ProductResult(**deepcopy({
            "id": "product-1", "name": "Test product", "description": None,
            "category": None, "color": None, "style": None, "gender": None,
            "season": None, "year": None, "price": None,
            "source": ProductSource.DB, **overrides,
        }))

    return create


@pytest.fixture
def product_search_results_factory():
    def create(*, internal_results=(), external_results=(), **overrides):
        return ProductSearchResults(
            internal_results=deepcopy(list(internal_results)),
            external_results=deepcopy(list(external_results)),
            **overrides,
        )

    return create
