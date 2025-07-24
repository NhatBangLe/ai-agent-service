from enum import Enum

from pydantic import Field

from .. import Configuration


class EmbeddingsType(str, Enum):
    HUGGING_FACE = "hugging_face"
    GOOGLE_GENAI = "google_genai"


class EmbeddingsConfiguration(Configuration):
    """
    An interface for embedding model configuration classes
    """
    name: str = Field(description="An unique name to determine embedding functions.")
    model_name: str = Field(min_length=1)
    type: EmbeddingsType = Field(description="The type of embedding model to use.")
