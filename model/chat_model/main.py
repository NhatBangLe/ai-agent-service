from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"


class LLMConfiguration(BaseModel):
    """
    An interface for large language model configuration classes
    """
    provider: LLMProvider = Field(description="All subclasses must specify this attribute.")
    model_name: str
    max_tokens: Annotated[int, Field(ge=10)] = 1024
    max_retries: Annotated[int, Field(ge=1)] = 2
    temperature: Annotated[float, Field(ge=0.0, le=1.0)]
