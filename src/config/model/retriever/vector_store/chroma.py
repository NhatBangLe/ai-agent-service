import chromadb

from src.config.model.retriever.vector_store.main import VectorStoreConfiguration


class ChromaVSConfiguration(VectorStoreConfiguration):
    """
    """
    tenant: str = chromadb.DEFAULT_TENANT
    database: str = chromadb.DEFAULT_DATABASE
