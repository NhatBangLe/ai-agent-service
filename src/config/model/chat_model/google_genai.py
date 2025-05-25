from typing import Literal

from pydantic import Field

from src.config.model.chat_model.main import LLMConfiguration


class GoogleGenAILLMConfiguration(LLMConfiguration):
    provider: Literal["google_genai"] = "google_genai"
    temperature: float = Field(
        description="Run inference with this temperature.", default=0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(
        description="Denotes the number of tokens to predict per generation.",
        default=1024, ge=10)
    max_retries: int = Field(
        description="Number of retries allowed for requests sent to the Anthropic Completion API.",
        default=6, ge=0)
    timeout: float | None = Field(description="Timeout for requests.", default=None, ge=0.0)
    top_k: int | None = Field(
        description="Decode using top-k sampling: consider the set of top_k most probable tokens.",
        default=None, ge=0)
    top_p: float | None = Field(
        description="Decode using nucleus sampling: consider the smallest set of tokens whose probability sum is at least top_p.",
        default=None, ge=0.0, le=1.0)
    # transport: Literal["rest", "grpc", "grpc_asyncio"] = Field(default="rest")
    # convert_system_message_to_human: bool = Field(
    #     description="Gemini does not support system messages; any unsupported messages will raise an error.",
    #     default=False)
