from typing import Dict, Optional, Annotated, Literal

from pydantic import BaseModel, Field

from src.model.embeddings.main import EmbeddingsModelConfiguration
from src.model.retriever.main import RetrieverConfiguration

DEFAULT_PERSIST_DIRECTORY = "./langchain_db"

VectorStoreConfigurationMode = Literal['persistent', 'remote']


class VectorStoreConnection(BaseModel):
    """
    Vector store connection class for the retriever.vector_store.connection property in configuration files.
    """
    host: str = "localhost"
    port: Annotated[int, Field(default=8000, gt=0)]
    ssl: bool = False
    headers: Dict[str, str] = None


class VectorStoreConfiguration(RetrieverConfiguration):
    """
    An interface for vector store configuration classes
    """
    mode: VectorStoreConfigurationMode = Field(default="persistent")
    connection: Optional[Annotated[VectorStoreConnection, Field(
        default=None,
        description="Connection will be used if the mode value is remote")]]
    collection_name: str
    embeddings_model: EmbeddingsModelConfiguration
    search_type: Literal['similarity', 'mmr', 'similarity_score_threshold'] = 'similarity'
    k: int = Field(default=4, description="Amount of documents to return")
    fetch_k: int = Field(default=20, description="Amount of documents to pass to MMR algorithm")
    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0, description="Diversity of results returned by MMR")
