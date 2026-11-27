from llm.repository.conversation_model_config_repository import ConversationModelConfigRepository
from llm.repository.embedding_cache_repository import EmbeddingCacheRepository


def get_embedding_cache_repo() -> EmbeddingCacheRepository:
    return EmbeddingCacheRepository()


def get_conversation_model_config_repo() -> ConversationModelConfigRepository:
    return ConversationModelConfigRepository()
