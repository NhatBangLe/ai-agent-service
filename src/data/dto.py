from datetime import datetime
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


class AssistantMessage(BaseModel):
    id: str
    role: str  # "user" | "assistant" | "system"
    content: str | list[dict[str, Any]]
    createdAt: datetime
    status: str | None = None  # "in_progress" | "done" | "error"


class ThreadMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ThreadCreate(BaseModel):
    if_exists: Literal["raise", "do_nothing"] = Field(
        default="raise",
        description="How to handle duplicate creation. "
                    "Must be either 'raise' (raise error if duplicate), or 'do_nothing' (return existing thread)"
    )


class AppendMessageRequest(BaseModel):
    parentId: str | None = None
    role: str = "user"
    content: str | list[dict[str, Any]]
