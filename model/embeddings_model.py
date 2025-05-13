from enum import Enum
from typing import Optional, Annotated

from pydantic import BaseModel, Field


class EmbeddingsModelProvider(str, Enum):
    HUGGING_FACE = "hugging_face"
    OLLAMA = "ollama"


class EmbeddingsModelConfiguration(BaseModel):
    """
    Embeddings model class for the embeddings_model property in configuration files.
    """
    provider: EmbeddingsModelProvider
    model_name: str
    api_token: Optional[Annotated[str, Field(default=None, description="Provide token as needed")]]
