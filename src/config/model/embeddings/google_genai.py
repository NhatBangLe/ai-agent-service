from enum import Enum

from pydantic import Field

from . import EmbeddingsConfiguration, EmbeddingsType


class GoogleGenAIEmbeddingsTaskType(str, Enum):
    UNSPECIFIED = "task_type_unspecified"
    RETRIEVAL_QUERY = "retrieval_query"
    RETRIEVAL_DOCUMENT = "retrieval_document"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class GoogleGenAIEmbeddingsConfiguration(EmbeddingsConfiguration):
    type: EmbeddingsType = Field(default=EmbeddingsType.GOOGLE_GENAI, frozen=True)
    task_type: GoogleGenAIEmbeddingsTaskType | None = Field(default=None)
