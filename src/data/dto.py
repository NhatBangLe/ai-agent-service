from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from .base_model import BaseImage, BaseLabel, BaseDocument


class LabelPublic(BaseLabel):
    id: int


class LabelCreate(BaseLabel):
    pass


class ImagePublic(BaseImage):
    id: UUID


class DocumentPublic(BaseDocument):
    id: UUID


class ThreadMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ThreadCreate(BaseModel):
    if_exists: Literal["raise", "do_nothing"] = Field(
        default="raise",
        description="How to handle duplicate creation. "
                    "Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing thread)"
    )
