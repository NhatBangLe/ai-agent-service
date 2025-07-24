from abc import abstractmethod, ABC
from enum import Enum

from pydantic import Field

from .. import Configuration


class EmbeddingsType(str, Enum):
    HUGGING_FACE = "hugging_face"
    GOOGLE_GENAI = "google_genai"


class EmbeddingsConfiguration(Configuration, ABC):
    """
    An interface for embedding model configuration classes
    """
    name: str = Field(description="An unique name to determine embedding functions.")
    model_name: str = Field(min_length=1)
    type: EmbeddingsType = Field(description="The type of embedding model to use.")

    @abstractmethod
    def get_api_key_env(self) -> str | None:
        pass
