from pydantic import field_validator

from src.config.model import Configuration
from src.config.model.chat_model import LLMConfiguration
from src.config.model.embeddings import EmbeddingsConfiguration
from src.config.model.prompt import PromptConfiguration
from src.config.model.recognizer.image import ImageRecognizerConfiguration
from src.config.model.retriever import RetrieverConfiguration
from src.config.model.tool import ToolConfiguration
from src.util.constant import SUPPORTED_LANGUAGE_DICT


# noinspection PyNestedDecorators
class AgentConfiguration(Configuration):
    """
    Agent configuration class for deserialize configuration files to pydantic object.
    """
    agent_name: str
    description: str | None = None
    language: str
    image_recognizer: ImageRecognizerConfiguration | None = None
    recognizer_output_config_path: str | None = None
    retrievers: list[RetrieverConfiguration] | None = None
    external_data_config_path: str | None = None
    embeddings: list[EmbeddingsConfiguration] | None = None
    tools: list[ToolConfiguration] | None = None
    llm: LLMConfiguration
    prompt: PromptConfiguration

    @field_validator("language", mode="after")
    @classmethod
    def validate_language(cls, v: str):
        if v not in SUPPORTED_LANGUAGE_DICT:
            raise ValueError(f'Unsupported {v} language.')
        return v
