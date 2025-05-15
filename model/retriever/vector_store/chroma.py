from typing import Optional

from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

from model.vector_store.main import VectorStoreConfiguration, VectorStoreConnection, VectorStoreProvider


class ChromaVSConnection(VectorStoreConnection):
    tenant: str = DEFAULT_TENANT
    database: str = DEFAULT_DATABASE


class ChromaVSConfiguration(VectorStoreConfiguration):
    provider = VectorStoreProvider.CHROMA
    settings: Optional[Settings] = None,
