from .base import EmbeddingConfig, BaseEmbeddingModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "None"):
    model_name = (embedding_model_name or "").lower()
    # OpenAI-compatible API embedding models (including Gemini gateway names)
    # should route to OpenAIEmbeddingModel, not local BGE.
    if "embedding" in model_name:
        from .OpenAI import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel

    if "bge-" in model_name or "nvidia/nv-embed" in model_name:
        from .BGEEmbedding import BGEEmbeddingModel

        return BGEEmbeddingModel

    logger.info(
        f"Unknown embedding model name: {embedding_model_name}, using BGEEmbeddingModel as default"
    )
    from .BGEEmbedding import BGEEmbeddingModel

    return BGEEmbeddingModel
