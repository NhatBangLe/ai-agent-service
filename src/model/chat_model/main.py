from pydantic import BaseModel, Field


class LLMConfiguration(BaseModel):
    """
    An interface for large language model configuration classes
    """
    model_name: str = Field(min_length=1)