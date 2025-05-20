from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE

from src.model.retriever.vector_store.main import VectorStoreConnection, VectorStoreConfiguration


class ChromaVSConnection(VectorStoreConnection):
    tenant: str = DEFAULT_TENANT
    database: str = DEFAULT_DATABASE


class ChromaVSConfiguration(VectorStoreConfiguration):
    """
    """
    # settings: Optional[Settings] = None,
