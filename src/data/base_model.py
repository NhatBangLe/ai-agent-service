import datetime

from sqlmodel import Field, SQLModel


class BaseLabel(SQLModel):
    name: str = Field(index=True, min_length=1, max_length=255, nullable=False, unique=True)
    description: str = Field(min_length=10, nullable=False)


class BaseFile(SQLModel):
    name: str = Field(index=True, min_length=1, max_length=255, nullable=False)
    mime_type: str = Field(max_length=100, nullable=False)
    created_at: datetime.datetime = Field(nullable=False)


class BaseImage(BaseFile):
    pass


class BaseDocument(BaseFile):
    pass


class BaseThread(SQLModel):
    title: str = Field(min_length=1, max_length=255, nullable=False)
    created_at: datetime.datetime = Field(nullable=False)
