from pydantic import Field

from model.chat_model.main import LLMConfiguration


class AnthropicLLMConfiguration(LLMConfiguration):
    api_key: str = Field(min_length=1)
    base_url: str = Field(min_length=1)