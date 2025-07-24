from enum import Enum

from pydantic import Field
from src.config.model import Configuration

__all__ = ["anthropic", "google_genai", "ollama", "ChatModelConfiguration", "ChatModelType"]


class ChatModelType(str, Enum):
    GOOGLE_GENAI = "google_genai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


class ChatModelConfiguration(Configuration):
    """
    An interface for large language model configuration classes
    """
    model_name: str = Field(min_length=1)
    type: ChatModelType = Field(description="The type of large language model to use.")
