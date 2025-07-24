from pydantic import Field

from . import EmbeddingsConfiguration, EmbeddingsType


class HuggingFaceEmbeddingsConfiguration(EmbeddingsConfiguration):
    """
    Embeddings model class for the embeddings_model property in configuration files.
    """

    def get_api_key_env(self) -> str | None:
        return None

    type: EmbeddingsType = Field(default=EmbeddingsType.HUGGING_FACE, frozen=True)
