from pydantic import Field, field_validator

from src.config.model.embeddings import EmbeddingsConfiguration
from src.config.model.retriever import RetrieverConfiguration


# noinspection PyNestedDecorators
class BM25Configuration(RetrieverConfiguration):
    embeddings_model: EmbeddingsConfiguration = Field(description="The embeddings configuration to use.")
    k: int = Field(description="Number of documents to return.", default=4)
    enable_remove_emoji: bool = Field(default=False)
    enable_remove_emoticon: bool = Field(default=False)
    removal_words_path: str | None = Field(default=None,
                                           description="Path to a word-file which provides removal words.")

    @field_validator("removal_words_path", mode="after")
    @classmethod
    def validate_removal_words_path(cls, value: str | None):
        if value is not None:
            if len(value) == 0 or len(value.strip()) == 0:
                return None
        return value
