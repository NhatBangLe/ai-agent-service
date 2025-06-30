from enum import Enum

from pydantic import Field

from src.config.model.embeddings import EmbeddingsConfiguration


class GoogleGenAIEmbeddingsTaskType(str, Enum):
    UNSPECIFIED = "task_type_unspecified"
    RETRIEVAL_QUERY = "retrieval_query"
    RETRIEVAL_DOCUMENT = "retrieval_document"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class GoogleGenAIEmbeddingsConfiguration(EmbeddingsConfiguration):
    task_type: GoogleGenAIEmbeddingsTaskType | None = Field(default=None)
