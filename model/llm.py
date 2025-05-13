from pydantic import BaseModel

from enum import Enum


class LLMProvider(str, Enum):
    HUGGING_FACE = "hugging_face"
    OLLAMA = "ollama"


class LLMConfiguration(BaseModel):
    provider: LLMProvider
    host: str
    port: int