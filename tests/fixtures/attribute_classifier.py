import pytest


class RecordingAttributeClassifier:
    def __init__(self, *, result=(), select=None):
        self.result = list(result)
        self.select = select
        self.context = ""
        self.attributes = []

    def relevant_attributes(self, context, attributes):
        self.context = context
        self.attributes = list(attributes)
        return list(self.result if self.select is None else self.select(attributes))


@pytest.fixture
def attribute_classifier_factory():
    return RecordingAttributeClassifier
