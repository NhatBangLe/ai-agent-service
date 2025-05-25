from pydantic import BaseModel, Field


class RetrieverConfiguration(BaseModel):
    """
    An interface for retriever configuration classes
    """
    weight: float = Field(description="Retriever weight for combining results", ge=0.0, le=1.0)
