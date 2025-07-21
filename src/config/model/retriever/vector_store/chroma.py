from pydantic import Field

from src.config.model.retriever.vector_store import VectorStoreConfiguration


class ChromaVSConfiguration(VectorStoreConfiguration):
    tenant: str = Field(default="default_tenant", min_length=1)
    database: str = Field(default="default_database", min_length=1)
