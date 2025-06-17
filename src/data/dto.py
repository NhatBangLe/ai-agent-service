from typing import Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from .base_model import BaseImage, BaseLabel, BaseDocument, BaseThread


class LabelPublic(BaseLabel):
    id: int


class LabelCreate(BaseLabel):
    pass


class ImagePublic(BaseImage):
    id: UUID


class DocumentPublic(BaseDocument):
    id: UUID
    embedded_to_vs: str | None
    embedded_to_bm25: bool


class AttachmentPublic(BaseModel):
    id: str
    mime_type: str


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
