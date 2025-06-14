from pydantic import Field, field_validator

from src.config.model import Configuration

__all__ = ["RetrieverConfiguration", "bm25", "vector_store"]


# noinspection PyNestedDecorators
class RetrieverConfiguration(Configuration):
    """
    An interface for retriever configuration classes
    """
    name: str = Field(description="An unique name is used for determining retrievers.")
    weight: float = Field(description="Retriever weight for combining results", ge=0.0, le=1.0)

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, name: str):
        if len(name.strip()) == 0:
            raise ValueError(f'Retriever name cannot be blank.')
        return name
