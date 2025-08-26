import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy.orm import RelationshipProperty
from sqlmodel import Field, SQLModel, Relationship

from .base_model import BaseImage, BaseLabel, BaseDocument, BaseThread, BaseFile
from ..util.function import get_datetime_now


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_threads: list["Thread"] = Relationship(back_populates="user")


class Thread(BaseThread, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", nullable=False)
    user: User = Relationship(back_populates="created_threads")
    attachments: list["File"] = Relationship(back_populates="thread")


class File(BaseFile, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    save_path: str = Field(nullable=False)
    image: Optional["Image"] = Relationship(back_populates="file",
                                            sa_relationship=RelationshipProperty(uselist=False,
                                                                                 viewonly=True))
    document: Optional["Document"] = Relationship(back_populates="file",
                                                  sa_relationship=RelationshipProperty(uselist=False,
                                                                                       viewonly=True))
    thread_id: UUID | None = Field(default=None, foreign_key="thread.id")
    thread: Thread | None = Relationship(back_populates="attachments")


class Label(BaseLabel, table=True):
    id: int | None = Field(ge=0, default=None, primary_key=True)
    labeled_images: list["LabeledImage"] = Relationship(back_populates="label", cascade_delete=True)
    classified_images: list["ClassifiedImage"] = Relationship(back_populates="label", cascade_delete=True)


class Image(BaseImage, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    assigned_labels: list["LabeledImage"] = Relationship(back_populates="image",
                                                         sa_relationship=RelationshipProperty(
                                                             lazy="selectin",
                                                             cascade="delete-orphan, save-update, delete"))
    classified_labels: list["ClassifiedImage"] = Relationship(back_populates="image",
                                                              sa_relationship=RelationshipProperty(
                                                                  lazy="selectin",
                                                                  cascade="delete-orphan, save-update, delete"))
    file_id: UUID = Field(description="Image file", index=True, foreign_key="file.id", nullable=False)
    file: File = Relationship(back_populates="image")


class LabeledImage(SQLModel, table=True):
    label_id: int = Field(foreign_key="label.id", primary_key=True)
    image_id: UUID = Field(foreign_key="image.id", primary_key=True)
    created_at: datetime.datetime = Field(nullable=False, default_factory=get_datetime_now)

    label: Label = Relationship(back_populates="labeled_images")
    image: Image = Relationship(back_populates="assigned_labels")


class ClassifiedImage(SQLModel, table=True):
    label_id: int = Field(foreign_key="label.id", primary_key=True)
    image_id: UUID = Field(foreign_key="image.id", primary_key=True)

    label: Label = Relationship(back_populates="classified_images")
    image: Image = Relationship(back_populates="classified_labels")


class Document(BaseDocument, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    embed_to_vs: str | None = Field(description="Name of the vector store that document is embedded to",
                                    default=None, nullable=True, max_length=100)
    chunks: list["DocumentChunk"] = Relationship(back_populates="document",
                                                 sa_relationship=RelationshipProperty(
                                                     lazy="selectin",
                                                     cascade="delete-orphan, save-update, delete"))
    file_id: UUID = Field(description="Document file", foreign_key="file.id", nullable=False)
    file: File = Relationship(back_populates="document")


class DocumentChunk(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True, max_length=36)
    document_id: UUID = Field(foreign_key="document.id", nullable=False)
    document: Document = Relationship(back_populates="chunks")
