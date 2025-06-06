from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .base_model import BaseImage, BaseLabel, BaseDocument
from ..util.function import strict_uuid_parser


class LabelPublic(BaseLabel):
    id: int


class LabelCreate(BaseLabel):
    pass


class ImagePublic(BaseImage):
    id: UUID


class DocumentPublic(BaseDocument):
    id: UUID


# noinspection PyNestedDecorators
class AttachmentPublic(BaseModel):
    image_id: str
    mime_type: str

    @field_validator("image_id", mode="after")
    @classmethod
    def validate_image_id(cls, v: str):
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
