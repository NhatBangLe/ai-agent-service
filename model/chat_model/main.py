from pydantic import BaseModel, Field


class LLMConfiguration(BaseModel):
    """
    An interface for large language model configuration classes
    """
    model_name: str = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=10)
    max_retries: int = Field(default=2, ge=1)
    temperature: float = Field(default=0.5, ge=0.0, le=1.0)
