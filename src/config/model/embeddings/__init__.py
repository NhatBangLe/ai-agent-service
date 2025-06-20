from pydantic import Field

from src.config.model import Configuration

__all__ = ["EmbeddingsConfiguration", "hugging_face"]


class EmbeddingsConfiguration(Configuration):
    """
    An interface for embeddings model configuration classes
    """
    model_name: str = Field(min_length=1)
