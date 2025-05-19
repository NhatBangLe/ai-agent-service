from typing import Literal

from pydantic import Field

from src.model.chat_model.main import LLMConfiguration


class AnthropicLLMConfiguration(LLMConfiguration):
    provider: Literal["anthropic"] = "anthropic"
    base_url: str = Field(
        description="Base URL for API requests. Only specify if using a proxy or service emulator.",
        default="https://api.anthropic.com", min_length=1)
    temperature: float = Field(
        description="A non-negative float that tunes the degree of randomness in generation.",
        default=0.5, ge=0.0, le=1.0)
    max_tokens: int = Field(
        description="Denotes the number of tokens to predict per generation.",
        default=1024, ge=10)
    max_retries: int = Field(
        description="Number of retries allowed for requests sent to the Anthropic Completion API.",
        default=2, ge=0)
    timeout: float | None = Field(description="Timeout for requests.", default=None, ge=0.0)
    stop_sequences: list[str] | None = Field(
        description="Default stop sequences.",
        default=None)
    top_k: int | None = Field(
        description="Number of most likely tokens to consider at each step.",
        default=None, ge=0.0)
    top_p: float | None = Field(
        description="Total probability mass of tokens to consider at each step.",
        default=None, ge=0.0)
