from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel, Relationship

from .base_model import BaseImage, BaseLabel


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    uploaded_images: list["Image"] = Relationship(back_populates="user")


class LabeledImage(SQLModel, table=True):
    image_id: UUID | None = Field(default_factory=uuid4, foreign_key="image.id", primary_key=True)
    label_id: int | None = Field(default=None, foreign_key="label.id", primary_key=True)


class Label(BaseLabel, table=True):
    id: int | None = Field(ge=0, default=None, primary_key=True)
    labeled_images: list["Image"] = Relationship(back_populates="has_labels", link_model=LabeledImage)


class Image(BaseImage, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    save_path: str = Field(nullable=False)
    has_labels: list["Label"] = Relationship(back_populates="labeled_images", link_model=LabeledImage)
    user_id: UUID = Field(description="Who uploaded this image", foreign_key="user.id")
    user: User = Relationship(back_populates="uploaded_images")
