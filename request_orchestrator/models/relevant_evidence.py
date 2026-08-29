from uuid import UUID

from pydantic import RootModel


class RelevantEvidenceByTool(RootModel[dict[str, list[UUID]]]):
    """Evaluator-selected evidence IDs grouped by the tool that produced them."""

    @classmethod
    def empty(cls) -> "RelevantEvidenceByTool":
        return cls({})
