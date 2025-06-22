from pydantic import BaseModel, Field

from src.config.model import Configuration


class ExternalDocument(BaseModel):
    name: str = Field(default="Unknown document", max_length=255)
    chunk_ids: list[str]


class VectorStoreContainsDocument(BaseModel):
    name: str = Field(description="An unique name of configured vector store.")
    documents: list[ExternalDocument]


class ExternalDocumentConfiguration(Configuration):
    version: str | None = Field(default=None)
    vector_stores: list[VectorStoreContainsDocument]
