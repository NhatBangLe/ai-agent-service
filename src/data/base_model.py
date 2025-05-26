from sqlmodel import Field, SQLModel


class BaseLabel(SQLModel):
    name: str = Field(index=True, min_length=1, max_length=255, nullable=False)
    description: str = Field(min_length=10, nullable=False)


class BaseImage(SQLModel):
    name: str = Field(index=True, min_length=1, max_length=255, nullable=False)
    mime_type: str = Field(max_length=100, nullable=False)
