from enum import Enum

from pydantic import BaseModel, Field


class RetrieverType(str, Enum):
    """
    All supported retrievers.
    """
    BM25 = "bm25"
    VECTOR_STORE = "vector_store"
    EXTERNAL_SEARCHING = "external_searching"


class RetrieverConfiguration(BaseModel):
    """
    An interface for retriever configuration classes
    """
    type: RetrieverType = Field(description="All subclasses must specify this attribute.")
    weight: float
