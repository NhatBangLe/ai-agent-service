from pydantic import Field

from ..chat_model import ChatModelConfiguration, ChatModelType


class OllamaChatModelConfiguration(ChatModelConfiguration):
    type: ChatModelType = Field(default=ChatModelType.OLLAMA, frozen=True)
    temperature: float = Field(
        description="The temperature of the model. Increasing the temperature will make the model answer more creatively.",
        default=0.8, ge=0.0, le=1.0)
    base_url: str | None = Field(
        description="Base url the model is hosted under.",
        default=None, min_length=1)
    seed: int | None = Field(
        description="Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt.",
        default=None)
    num_ctx: int = Field(
        description="Sets the size of the context window used to generate the next token.",
        default=2048)
    num_predict: int | None = Field(
        description="Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)",
        default=128)
    repeat_penalty: float | None = Field(
        description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
        default=1.1)
    top_k: int | None = Field(
        description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.",
        default=40, ge=0)
    top_p: float | None = Field(
        description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
        default=0.9, ge=0.0, le=1.0)
    stop: list[str] | None = Field(
        description="Sets the stop tokens to use.",
        default=None)
