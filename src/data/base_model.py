import datetime
from enum import Enum

from sqlmodel import Field, SQLModel


class LabelSource(Enum):
    PREDEFINED = "PREDEFINED"
    CREATED = "CREATED"


class BaseLabel(SQLModel):
    name: str = Field(index=True, min_length=1, max_length=255, nullable=False, unique=True)
    description: str = Field(max_length=255)
    source: LabelSource = Field(description="Source of the label")


class BaseFile(SQLModel):
    name: str = Field(index=True, min_length=1, max_length=255, nullable=False)
    created_at: datetime.datetime = Field(nullable=False)


class BaseImage(BaseFile):
    mime_type: str = Field(max_length=100, nullable=False)


class DocumentSource(Enum):
    UPLOADED = "UPLOADED"
    EXTERNAL = "EXTERNAL"


class BaseDocument(BaseFile):
    mime_type: str | None = Field(max_length=100, nullable=True)
    description: str | None = Field(default=None, nullable=True, max_length=255)
    source: DocumentSource = Field(nullable=False)


class BaseThread(SQLModel):
    title: str = Field(min_length=1, max_length=255, nullable=False)
    created_at: datetime.datetime = Field(nullable=False)
