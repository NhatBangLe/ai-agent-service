import datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import RelationshipProperty
from sqlmodel import Field, SQLModel, Relationship

from .base_model import BaseImage, BaseLabel, BaseDocument, BaseThread, BaseFile
from ..util.function import get_datetime_now


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    uploaded_images: list["Image"] = Relationship(back_populates="user")
    created_threads: list["Thread"] = Relationship(back_populates="user")


class Label(BaseLabel, table=True):
    id: int | None = Field(ge=0, default=None, primary_key=True)
    labeled_images: list["LabeledImage"] = Relationship(back_populates="label", cascade_delete=True)
    classified_images: list["ClassifiedImage"] = Relationship(back_populates="label", cascade_delete=True)


class File(BaseFile, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    save_path: str = Field(nullable=False)


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
    user_id: UUID = Field(description="Who uploaded this image", foreign_key="user.id", nullable=False)
    user: User = Relationship(back_populates="uploaded_images")
    file_id: str = Field(description="Uploaded file", nullable=False)


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
    embed_bm25: bool = Field(description="Whether this document is embedded to BM25 index", default=False)
    chunks: list["DocumentChunk"] = Relationship(back_populates="document",
                                                 sa_relationship=RelationshipProperty(
                                                     lazy="selectin",
                                                     cascade="delete-orphan, save-update, delete"))
    file_id: str | None = Field(description="Uploaded file", default=None, nullable=True)


class DocumentChunk(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True, max_length=36)
    document_id: UUID = Field(foreign_key="document.id", nullable=False)
    document: Document = Relationship(back_populates="chunks")


class Thread(BaseThread, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", nullable=False)
    user: User = Relationship(back_populates="created_threads")
