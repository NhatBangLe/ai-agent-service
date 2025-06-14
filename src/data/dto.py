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


class PagingWrapper[T](BaseModel):
    """
    The `PagingWrapper` class provides a standardized structure for encapsulating
    paginated results from an API or database query. It inherits from `BaseModel`
    for data validation and serialization, and uses `Generic[T]` to allow for
    flexible content types.
    """

    content: list[T] = Field(description="Return content")
    first: bool | None = Field(default=None, description="Whether this is a first page.")
    last: bool | None = Field(default=None, description="Whether this is a last page.")
    page_number: int = Field(description="The page number.")
    page_size: int = Field(description="The page size.")
    total_elements: int | None = Field(default=None, description="The total number of elements in database.")
    total_pages: int | None = Field(default=None,
                                    description="The total number of pages in database if use `page_size`.")


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
