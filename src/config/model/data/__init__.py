from pydantic import BaseModel, Field

from src.config.model import Configuration


class ExternalDocument(BaseModel):
    name: str = Field(default="Unknown document", max_length=255)
    chunk_ids: list[str]


class ExternalDocumentConfiguration(Configuration):
    version: str | None = Field(default=None)
    is_configured: bool = Field(default=False)
    documents: list[ExternalDocument]
