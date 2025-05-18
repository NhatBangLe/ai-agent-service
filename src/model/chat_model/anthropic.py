from pydantic import Field

from src.model.chat_model.main import LLMConfiguration


class AnthropicLLMConfiguration(LLMConfiguration):
    base_url: str | None = Field(default=None, min_length=1)