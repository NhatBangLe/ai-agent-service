from typing import Annotated
from pydantic import Field

from model.chat_model.main import LLMConfiguration, LLMProvider


class AnthropicLLMConfiguration(LLMConfiguration):
    type: LLMProvider.ANTHROPIC
    api_key: Annotated[str, Field(alias="anthropic_api_key")]
    base_url: Annotated[str, Field(alias="anthropic_api_url")]