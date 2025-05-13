from enum import Enum
from typing import Any, Dict, Optional, Annotated
from pydantic import BaseModel, Field

from model.embeddings_model import EmbeddingsModelConfiguration


class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"


class VectorStoreConfigurationMode(str, Enum):
    PERSISTENT = "persistent"
    REMOTE = "remote"


class VectorStoreConnection(BaseModel):
    """
    Vector store connection class for the retriever.vector_store.connection property in configuration files.
    """
    host: str = "localhost"
    port: Annotated[int, Field(default=8000, gt=0)]
    ssl: bool = False
    headers: Dict[str, str] = None
    additional_settings: Optional[Dict[str, Any]] = None


class VectorStoreConfiguration(BaseModel):
    """
    Vector store class for the retriever.vector_store property in configuration files.
    """
    provider: VectorStoreProvider
    mode: VectorStoreConfigurationMode
    connection: Optional[Annotated[VectorStoreConnection, Field(
        default=None,
        description="Connection will be used if the mode value is remote")]]
    collection_name: str
    embeddings_model: EmbeddingsModelConfiguration

