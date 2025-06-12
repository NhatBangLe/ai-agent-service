from typing import Any, Self

from pydantic import BaseModel, Field, field_validator


# noinspection PyNestedDecorators
class RetrieverConfiguration(BaseModel):
    """
    An interface for retriever configuration classes
    """
    name: str = Field(description="An unique name is used for determining retrievers.")
    weight: float = Field(description="Retriever weight for combining results", ge=0.0, le=1.0)

    @field_validator("classes", mode="after")
    @classmethod
    def validate_name(cls, name: str):
        if len(name.strip()) == 0:
            raise ValueError(f'Retriever name cannot be blank.')
        return name
