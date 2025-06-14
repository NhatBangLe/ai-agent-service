from enum import Enum
from typing import Literal

from langchain_google_genai import HarmCategory as GenAIHarmCategory, HarmBlockThreshold as GenAIHarmBlockThreshold
from pydantic import Field

from src.config.model.chat_model.main import LLMConfiguration


class HarmCategory(Enum):
    UNSPECIFIED = "UNSPECIFIED"
    DEROGATORY = "DEROGATORY"
    TOXICITY = "TOXICITY"
    VIOLENCE = "VIOLENCE"
    SEXUAL = "SEXUAL"
    MEDICAL = "MEDICAL"
    DANGEROUS = "DANGEROUS"
    HARASSMENT = "HARASSMENT"
    HATE_SPEECH = "HATE_SPEECH"
    SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
    DANGEROUS_CONTENT = "DANGEROUS_CONTENT"
    CIVIC_INTEGRITY = "CIVIC_INTEGRITY"


class HarmBlockThreshold(Enum):
    UNSPECIFIED = "UNSPECIFIED"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_NONE = "BLOCK_NONE"
    OFF = "OFF"


HARM_CATEGORY_DICT = {
    "UNSPECIFIED": GenAIHarmCategory.HARM_CATEGORY_UNSPECIFIED,
    "DEROGATORY": GenAIHarmCategory.HARM_CATEGORY_DEROGATORY,
    "TOXICITY": GenAIHarmCategory.HARM_CATEGORY_TOXICITY,
    "VIOLENCE": GenAIHarmCategory.HARM_CATEGORY_VIOLENCE,
    "SEXUAL": GenAIHarmCategory.HARM_CATEGORY_SEXUAL,
    "MEDICAL": GenAIHarmCategory.HARM_CATEGORY_MEDICAL,
    "DANGEROUS": GenAIHarmCategory.HARM_CATEGORY_DANGEROUS,
    "HARASSMENT": GenAIHarmCategory.HARM_CATEGORY_HARASSMENT,
    "HATE_SPEECH": GenAIHarmCategory.HARM_CATEGORY_HATE_SPEECH,
    "SEXUALLY_EXPLICIT": GenAIHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    "DANGEROUS_CONTENT": GenAIHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    "CIVIC_INTEGRITY": GenAIHarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
}

HARM_BLOCK_THRESHOLD_DICT = {
    "UNSPECIFIED": GenAIHarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,
    "BLOCK_LOW_AND_ABOVE": GenAIHarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    "BLOCK_MEDIUM_AND_ABOVE": GenAIHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "BLOCK_ONLY_HIGH": GenAIHarmBlockThreshold.BLOCK_ONLY_HIGH,
    "BLOCK_NONE": GenAIHarmBlockThreshold.BLOCK_NONE,
    "OFF": GenAIHarmBlockThreshold.OFF,
}


def convert_safety_settings_to_genai(settings: dict[str, str]):
    result: dict[GenAIHarmCategory, GenAIHarmBlockThreshold] = {}
    for k, v in settings.items():
        result[HARM_CATEGORY_DICT[k]] = HARM_BLOCK_THRESHOLD_DICT[v]
    return result


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
    safety_settings: dict[str, str] | None = Field(
        default=None,
        description="The default safety settings to use for all generations.")
    # transport: Literal["rest", "grpc", "grpc_asyncio"] = Field(default="rest")
    # convert_system_message_to_human: bool = Field(
    #     description="Gemini does not support system messages; any unsupported messages will raise an error.",
    #     default=False)
