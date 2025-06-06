from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .base_model import BaseImage, BaseLabel, BaseDocument


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


# noinspection PyNestedDecorators
class AttachmentPublic(BaseModel):
    image_id: str
    mime_type: str

    @field_validator("image_id", mode="after")
    @classmethod
    def validate_image_id(cls, v: str):
        from ..util.function import strict_uuid_parser
        strict_uuid_parser(v)
        return v


class InputMessage(BaseModel):
    attachments: list[AttachmentPublic] | None
    content: str


class ThreadCreate(BaseModel):
    if_exists: Literal["raise", "do_nothing"] = Field(
        default="raise",
        description="How to handle duplicate creation. "
                    "Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing thread)"
    )
