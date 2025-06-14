from pydantic import BaseModel, Field

__all__ = ["EmbeddingsModelConfiguration", "hugging_face"]


class EmbeddingsModelConfiguration(BaseModel):
    """
    An interface for embeddings model configuration classes
    """
    model_name: str = Field(min_length=1)
