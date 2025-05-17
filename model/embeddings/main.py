from pydantic import BaseModel, Field


class EmbeddingsModelConfiguration(BaseModel):
    """
    An interface for embeddings model configuration classes
    """
    model_name: str = Field(min_length=1)
