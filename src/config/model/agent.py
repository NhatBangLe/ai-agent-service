from pydantic import field_validator, Field

from src.config.model import Configuration
from src.config.model.chat_model import ChatModelConfiguration
from src.config.model.mcp import MCPConfiguration
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
    agent_name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=255)
    language: str
    image_recognizer: ImageRecognizerConfiguration | None = Field(default=None)
    retrievers: list[RetrieverConfiguration] | None = Field(default=None)
    tools: list[ToolConfiguration] | None = Field(default=None)
    mcp: MCPConfiguration | None = Field(default=None)
    llm: ChatModelConfiguration
    prompt: PromptConfiguration

    @field_validator("language", mode="after")
    @classmethod
    def validate_language(cls, v: str):
        if v not in SUPPORTED_LANGUAGE_DICT:
            raise ValueError(f'Unsupported {v} language.')
        return v
