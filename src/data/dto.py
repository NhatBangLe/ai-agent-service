from typing import Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from .base_model import BaseImage, BaseLabel, BaseDocument, BaseThread


class LabelPublic(BaseLabel):
    id: int


class LabelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


class LabelDelete(BaseModel):
    id: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, min_length=1)


class ImagePublic(BaseImage):
    id: UUID


class DocumentPublic(BaseDocument):
    id: UUID
    embedded_to_vs: str | None
    embedded_to_bm25: bool


class AttachmentPublic(BaseModel):
    id: str


class InputMessage(BaseModel):
    attachments: Sequence[AttachmentPublic] | None
    content: str


class OutputMessage(BaseModel):
    id: str | None = Field(default=None)
    content: str
    role: Literal["Human", "AI"]


class ThreadPublic(BaseThread):
    id: UUID


class ThreadCreate(BaseModel):
    title: str = Field()
