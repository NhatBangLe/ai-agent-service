import datetime
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel, Relationship

from .base_model import BaseImage, BaseLabel, BaseDocument, BaseThread, BaseFile


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    uploaded_images: list["Image"] = Relationship(back_populates="user")
    created_threads: list["Thread"] = Relationship(back_populates="user")


class Label(BaseLabel, table=True):
    id: int | None = Field(ge=0, default=None, primary_key=True)
    labeled_images: list["LabeledImage"] = Relationship(back_populates="label", cascade_delete=True)


class File(BaseFile, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    save_path: str = Field(nullable=False)
    uploaded_image: "Image" = Relationship(back_populates="file")
    uploaded_document: "Document" = Relationship(back_populates="file")


class Image(BaseImage, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    has_labels: list["LabeledImage"] = Relationship(back_populates="image", cascade_delete=True)
    user_id: UUID = Field(description="Who uploaded this image", foreign_key="user.id", nullable=False)
    user: User = Relationship(back_populates="uploaded_images")
    file_id: UUID = Field(description="Uploaded file", foreign_key="file.id", nullable=False)
    file: File = Relationship(back_populates="uploaded_image")


class LabeledImage(SQLModel, table=True):
    label_id: int = Field(foreign_key="label.id", primary_key=True)
    image_id: UUID = Field(foreign_key="image.id", primary_key=True)
    created_at: datetime.datetime = Field(nullable=False)

    label: Label = Relationship(back_populates="labeled_images")
    image: Image = Relationship(back_populates="has_labels")


class Document(BaseDocument, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    embed_to_vs: str | None = Field(description="Name of the vector store that document is embedded to",
                                    default=None, nullable=True, max_length=100)
    embed_bm25: bool = Field(description="Whether this document is embedded to BM25 index", default=False)
    chunks: list["DocumentChunk"] = Relationship(back_populates="document", cascade_delete=True)
    file_id: UUID | None = Field(description="Uploaded file", default=None, foreign_key="file.id", nullable=True)
    file: File | None = Relationship(back_populates="uploaded_document")


class DocumentChunk(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True, max_length=36)
    document_id: UUID = Field(foreign_key="document.id", nullable=False)
    document: Document = Relationship(back_populates="chunks")


class Thread(BaseThread, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", nullable=False)
    user: User = Relationship(back_populates="created_threads")
