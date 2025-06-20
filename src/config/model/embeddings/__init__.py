from pydantic import Field

from src.config.model import Configuration

__all__ = ["EmbeddingsConfiguration", "hugging_face", "google_genai"]


class EmbeddingsConfiguration(Configuration):
    """
    An interface for embedding model configuration classes
    """
    name: str = Field(description="An unique name to determine embedding functions.")
    model_name: str = Field(min_length=1)
