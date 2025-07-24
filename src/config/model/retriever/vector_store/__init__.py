from abc import ABC
from typing import Literal

from pydantic import BaseModel, Field

from src.config.model.embeddings import EmbeddingsConfiguration
from src.config.model.retriever import RetrieverConfiguration

DEFAULT_COLLECTION_NAME = "agent_collection"
VectorStoreConfigurationMode = Literal['persistent', 'remote']


class VectorStoreConnection(BaseModel):
    """
    Vector store connection class for the retriever.vector_store.connection property in configuration files.
    """
    host: str = "localhost"
    port: int = Field(default=8000, gt=0)
    ssl: bool = False
    headers: dict[str, str] | None = None


class VectorStoreConfiguration(RetrieverConfiguration, ABC):
    """
    An interface for vector store configuration classes
    """
    mode: VectorStoreConfigurationMode = Field(default="persistent")
    external_data_config_path: str | None = Field(default=None)
    connection: VectorStoreConnection | None = Field(
        default=None, description="Connection will be used if the mode value is remote")
    collection_name: str = Field(default=DEFAULT_COLLECTION_NAME)
    embeddings_model: EmbeddingsConfiguration = Field(description="The embeddings configuration to use.")
    search_type: Literal['similarity', 'mmr'] = 'similarity'
    k: int = Field(default=4, description="Amount of documents to return")
    fetch_k: int = Field(default=20, description="Amount of documents to pass to MMR algorithm")
    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0, description="Diversity of results returned by MMR")
