from enum import Enum

from pydantic import BaseModel, Field


class EmbeddingsModelProvider(str, Enum):
    HUGGING_FACE = "hugging_face"
    OLLAMA = "ollama"


class EmbeddingsModelConfiguration(BaseModel):
    """
    An interface for embeddings model configuration classes
    """
    provider: EmbeddingsModelProvider = Field(description="All subclasses must specify this attribute.")
    model_name: str
